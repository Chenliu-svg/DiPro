import torch
import torch.nn as nn

from models.Base_Predictor import Base_Predictor
from models.Component import LearnableRegionPadding, LongitudinalCXRModel, GroupEmbedding, MultisacleMultimodalFusion
from mimic_data_extract.mimic3benchmark.constants import SELECTED_REGIONS, TASK_NUM_CLASSES, FILL_VALUE, TASK_OUTPUT_SIZE, IMAGE_CHANNEL, IMAGE_SIZE,  VARIABLE_DIM,NUM_DEMOGRAPHIC_VAR

class DiPro(Base_Predictor):
    def __init__(self, 
                # task related parameters
                task_name,
                label_path, 

                # different loss weights
                 lambda_consistency,  
                 lambda_orthogonal, 
                 lambda_static_order, 
                 lambda_dynamic_order,
                 lambda_CE_loss,

                # other hyperparameters
                 num_transformer_layers, 
                 group_num,
                 d_model, 
                 dropout_rate, 
                num_fusion_layers=1, 
                freeze_backbone=False,
                mode='max', 
                max_epochs=100, 
                ckpt_path=None):
        
        # initialization of the base predictor
        super().__init__(task_name=task_name, label_path=label_path, mode=mode, max_epochs=max_epochs, ckpt_path=ckpt_path)
        
        # different loss weights
        self.lambda_consistency=lambda_consistency
        self.lambda_static_order=lambda_static_order
        self.lambda_dynamic_order=lambda_dynamic_order
        self.lambda_orthogonal = lambda_orthogonal
        self.lambda_CE_loss = lambda_CE_loss


        self.output_size=TASK_OUTPUT_SIZE[self.task_name]
        self.num_final_class=TASK_NUM_CLASSES[self.task_name]

        # learnable padding for missing regions
        num_selected_region=len(SELECTED_REGIONS)
        self.learnable_pad=LearnableRegionPadding((num_selected_region,IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

       
        # The STD and PAE modules handling the longitudinal CXRs
        if self.task_name=='disease_progression':
            self.long_cxr_encoder = LongitudinalCXRModel(loss_ls=self.loss, device=self.device, feature_size=d_model, dropout_rate=dropout_rate, freeze_backbone=freeze_backbone)
        else:
            assert self.task_name in ['mortality', 'length_of_stay']
            self.long_cxr_encoder = LongitudinalCXRModel(loss_ls=self.loss_ls, device=self.device, feature_size=d_model, dropout_rate=dropout_rate, freeze_backbone=freeze_backbone)
        
        
        self.long_multi_modal_fuison=MultisacleMultimodalFusion(num_final_class=self.num_final_class, ehr_num_var=VARIABLE_DIM,d_model=d_model, num_layers=num_transformer_layers, dropout_rate=dropout_rate, num_fusion_layers=num_fusion_layers)
    
        # group embedding
        self.group_embedding = GroupEmbedding(D=d_model, K=group_num)

        # demographic encoder
        # [batch_size, NUM_DEMOGRAPHIC_VAR]->[batch_size, NUM_DEMOGRAPHIC_VAR, d_model]
        self.demographic_encoder = nn.Sequential(
            nn.Linear(NUM_DEMOGRAPHIC_VAR, d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # final fusion with demographic 
        self.Demographic_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.Demographic_norm = nn.LayerNorm(d_model)

        self.output_layer = nn.ModuleList()

        # self.symp_prog_fc=nn.Linear(self.symp_embedding_dim, self.output_nc)
        for cls_ in range(self.num_final_class):
            self.output_layer.append(nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, self.output_size)
        ))

    
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)


    def forward(self,reshaped_ehr, reshaped_ehr_mask, reshaped_region, reshaped_region_mask, demograhic_ehr, reshaped_y_aux, ehr_times, interval_times):
        # repad the reshaped_region using learnable padding according to the reshaped_region_mask
        with torch.cuda.amp.autocast():
            reshaped_region = self.learnable_pad(reshaped_region, reshaped_region_mask)
        
        # Modeling longitudinal CXRs
        static_cxr, dynamic_cxr, consistency_loss, orthogonal_loss , static_loss, dynamic_loss = self.long_cxr_encoder(reshaped_region,reshaped_y_aux)
        
        # [batch_size, num_pairs, num_regions, d_model] -> [batch_size, num_pairs*num_regions, d_model]
        static_cxr = static_cxr.view(static_cxr.shape[0], -1, static_cxr.shape[-1])
        
        
        # encode demographic data
        demographic = self.demographic_encoder(demograhic_ehr).unsqueeze(1)

        # concatenate demographic and static cxr features
        demo_static=torch.cat([static_cxr,demographic],dim=1)
        encoded_demo_static=self.group_embedding(demo_static)

    
        # multiscale multimodal fusion
        reshaped_region_mask = reshaped_region_mask[:, 1:]
        reshaped_region_mask = reshaped_region_mask.bool()
        fused_cxr_ehr=self.long_multi_modal_fuison(dynamic_cxr, reshaped_region_mask, reshaped_ehr, reshaped_ehr_mask, ehr_times, interval_times)
        
        # Final static fusion
        attn_out2, _ = self.Demographic_attn(
            query=fused_cxr_ehr,  # [B, num_final_class+num_cxr, D]
            key=encoded_demo_static,            # [B, num_demo, D]
            value=encoded_demo_static           # [B, num_demo, D]
        )

        cls_final = self.Demographic_norm(fused_cxr_ehr + attn_out2) 
        cls_final = cls_final[:, :self.num_final_class, :]  # [B,num_final_class, D]
        
        # output
        output_logits=[]
        for cls_ in range(self.num_final_class):
            output = self.output_layer[cls_](cls_final[:,cls_,:])
            output_logits.append(output)
        output=torch.stack(output_logits,dim=1)
        
        return output, consistency_loss, orthogonal_loss , static_loss, dynamic_loss
    
    def training_step(self, batch, batch_idx):
        self.setup('train')
        reshaped_ehr, reshaped_ehr_mask, reshaped_region, reshaped_region_mask, demograhic_ehr, y, reshaped_y_aux, reshaped_y_aux_mask,  ehr_times, interval_times=batch
        
        output, consistency_loss, orthogonal_loss , static_loss, dynamic_loss = self(reshaped_ehr, reshaped_ehr_mask, reshaped_region, reshaped_region_mask, demograhic_ehr,reshaped_y_aux, ehr_times, interval_times)
        CE_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if self.task_name=='disease_progression':
            for i in range(self.num_final_class):
                disease_pred=output[:,i]
                gt=y[:,i]
                if torch.all((gt==FILL_VALUE)):
                    continue
                disease_pred=disease_pred[gt!=FILL_VALUE]
                gt=gt[gt!=FILL_VALUE]
                CE_loss=CE_loss+self.loss[i](disease_pred,gt)
        else: 
            CE_loss=self.loss(output[:,0].squeeze(1),y)
        
        
        loss = CE_loss*self.lambda_CE_loss + self.lambda_orthogonal*orthogonal_loss + self.lambda_consistency*consistency_loss+self.lambda_static_order*static_loss+self.lambda_dynamic_order*dynamic_loss
        self.log("loss", loss)
        self.log("orthogonal_loss", orthogonal_loss)
        self.log("CE_loss", CE_loss)
        self.log("consistency_loss", consistency_loss)
        self.log("static_loss",static_loss)
        self.log("dynamic_loss",dynamic_loss)

        return {"loss": loss, "orthogonal_loss":orthogonal_loss, "CE_loss":CE_loss, "consistency_loss":consistency_loss,"static_loss":static_loss, "dynamic_loss":dynamic_loss}




    def validation_step(self, batch, batch_idx):
        self.setup('validation')
        reshaped_ehr, reshaped_ehr_mask, reshaped_region, reshaped_region_mask, demograhic_ehr, y, reshaped_y_aux,_,  ehr_times, interval_times=batch
        output, consistency_loss, orthogonal_loss , static_loss, dynamic_loss= self(reshaped_ehr, reshaped_ehr_mask, reshaped_region, reshaped_region_mask, demograhic_ehr, reshaped_y_aux, ehr_times, interval_times)
        CE_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if self.task_name=='disease_progression':
            for i in range(self.num_final_class):
                disease_pred=output[:,i]
                gt=y[:,i]
                if torch.all((gt==FILL_VALUE)):
                    continue
                disease_pred=disease_pred[gt!=FILL_VALUE]
                gt=gt[gt!=FILL_VALUE]
                CE_loss=CE_loss+self.loss[i](disease_pred,gt)
        else: 
            CE_loss=self.loss(output[:,0].squeeze(1),y)

        return {'val_loss': CE_loss, 'target': y, 'preds': output}


    def test_step(self, batch, batch_idx):
        self.setup('test')
        reshaped_ehr, reshaped_ehr_mask, reshaped_region, reshaped_region_mask, demograhic_ehr, y, reshaped_y_aux,_,  ehr_times, interval_times=batch
        output, consistency_loss, orthogonal_loss , static_loss, dynamic_loss= self(reshaped_ehr, reshaped_ehr_mask, reshaped_region, reshaped_region_mask, demograhic_ehr, reshaped_y_aux, ehr_times, interval_times)
        CE_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if self.task_name=='disease_progression':
            for i in range(self.num_final_class):
                disease_pred=output[:,i]
                gt=y[:,i]
                if torch.all((gt==FILL_VALUE)):
                    continue
                disease_pred=disease_pred[gt!=FILL_VALUE]
                gt=gt[gt!=FILL_VALUE]
                CE_loss=CE_loss+self.loss[i](disease_pred,gt)
        else: 
            CE_loss=self.loss(output[:,0].squeeze(1),y)

        return {'test_loss': CE_loss, 'target': y, 'preds': output}



