
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import os
import numpy as np
from torch import nn

import pytorch_lightning as pl
from sklearn.metrics import f1_score,roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_recall_curve, auc

from mimic_data_extract.mimic3benchmark.constants import PROGRESSION_DICT, PROGRESSION_DISEASE, FILL_VALUE, TASK_OUTPUT_SIZE, TASK_MONITOR


class Base_Predictor(pl.LightningModule):
    
    def __init__(self,
                 task_name,
                 label_path,
                 mode='max',
                 max_epochs=100,
                 ckpt_path=None,
                 ):
        super().__init__()
        self.task_name=task_name
        self.label_path = label_path
        self.mode = mode
        self.max_epochs=max_epochs

        self.monitor=TASK_MONITOR[self.task_name]
        self.ckpt_path=ckpt_path

        self.setup('train')


    def setup(self,stage):
        if self.task_name=='disease_progression':
            self.loss=[]
            self.loss_batch=[]
            self.data_df = pd.read_csv(os.path.join(self.label_path,"disease_progression", 'train.csv'))
            for col in PROGRESSION_DISEASE:
                self.data_df[col] = self.data_df[col].map(PROGRESSION_DICT)

                progression=self.data_df[col]
                # calculate the weight of each class
                pos_weight=[progression.count().sum()/(len(progression[progression==i])*3) for i in range(3)]
                pos_weight=torch.tensor(pos_weight).to(self.device).float()
                
                # loss for final prediction
                loss = nn.CrossEntropyLoss(weight=pos_weight) 
                
                loss_batch=nn.CrossEntropyLoss(weight=pos_weight,reduction='none')
                self.loss.append(loss)
                self.loss_batch.append(loss_batch)
            
        else:
            # get loss weight for the PAE module
            self.loss_ls=[]
            self.data_df = pd.read_csv(os.path.join(self.label_path,"disease_progression", 'train.csv'))
            for col in PROGRESSION_DISEASE:
                self.data_df[col] = self.data_df[col].map(PROGRESSION_DICT)

                # breakpoint()
                progression=self.data_df[col]
                # calculate the weight of each class
                pos_weight=[progression.count().sum()/(len(progression[progression==i])*3) for i in range(3)]
                pos_weight=torch.tensor(pos_weight).to(self.device).float()

                loss = nn.CrossEntropyLoss(weight=pos_weight) 
                
                self.loss_ls.append(loss)

            # get loss for final prediction task
            assert self.task_name in ['mortality','length_of_stay']
            # /home/liuc/2025NIPS/fold0/task3_mortality_los
            self.data_df = pd.read_csv(os.path.join(self.label_path,"mortality_los", 'train.csv'))
            if self.task_name=='mortality':
                self.data_df = self.data_df['mortality_inhospital']
                pos_weight=[len(self.data_df[self.data_df==1])/len(self.data_df)]
                pos_weight=torch.tensor(pos_weight).to(self.device).float()
                self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                self.loss_batch=nn.BCEWithLogitsLoss(pos_weight=pos_weight,reduction='none')
            else:
                # los
                self.loss=nn.CrossEntropyLoss()
                self.loss_batch=nn.CrossEntropyLoss(reduction='none')
            
        self.num_classes=TASK_OUTPUT_SIZE[self.task_name]

        
    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
    
    def init_from_ckpt(self, path, ignore_keys=list()):

        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val/loss', avg_loss, prog_bar=True, on_epoch=True)


        all_target = torch.cat([x['target'] for x in outputs])
        
        all_preds = torch.cat([x['preds'] for x in outputs])
        
        if self.task_name == 'mortality':
            all_preds=all_preds.squeeze(1).squeeze(1)
            roc_auc, pr_auc = self.calculate_binary_metrics(all_preds, all_target)
            self.log('val/roc_auc', roc_auc, prog_bar=True, on_epoch=True)
            self.log('val/pr_auc', pr_auc, prog_bar=True, on_epoch=True)


        elif self.task_name=='disease_progression':
            total_metrics=self.calculate_multiclass_metrics(all_preds, all_target)
          
            for metric_name in total_metrics['atelectasis'].keys():
                values = [v[metric_name] for v in total_metrics.values()]
                mean_value = np.mean(values)
                self.log(f'val/{metric_name}', mean_value, prog_bar=True, on_epoch=True)
        
        else:
            assert self.task_name=='length_of_stay'
            all_preds=all_preds.squeeze(1).cpu()
            all_target=all_target.cpu()
            final_metrics=self.calculate_multiclass_for_one_class(all_preds, all_target)
            for k,v in final_metrics.items():
                self.log(f'val/{k}', v, prog_bar=True, on_epoch=True)
            
        del avg_loss,all_target,all_preds
    
    def test_epoch_end(self, outputs) -> None:
        logdir=self.trainer.logger.save_dir
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('test/loss', avg_loss, prog_bar=True, on_epoch=True)

        all_target = torch.cat([x['target'] for x in outputs])
       
        all_preds = torch.cat([x['preds'] for x in outputs])
       
        if self.task_name =='mortality':
            all_preds=all_preds.squeeze(1).squeeze(1)
            roc_auc, pr_auc = self.calculate_binary_metrics(all_preds, all_target)
            self.log('test/roc_auc', roc_auc, prog_bar=True, on_epoch=True)
            self.log('test/pr_auc', pr_auc, prog_bar=True, on_epoch=True)
            # save to file
            df_metrics=pd.DataFrame(data={'roc_auc':[roc_auc.item()],'pr_auc':[pr_auc.item()]})
            df_metrics.to_csv(os.path.join(logdir,'total_metrics.csv'))

        elif self.task_name=='disease_progression':
            total_metrics=self.calculate_multiclass_metrics(all_preds, all_target)
            mean_value_dict={}
            for metric_name in total_metrics['atelectasis'].keys():

                values = [v[metric_name] for v in total_metrics.values()]
                mean_value = np.mean(values)
                mean_value_dict[metric_name]=mean_value

                self.log(f'test/{metric_name}', mean_value, prog_bar=True, on_epoch=True)
            total_metrics['mean']=mean_value_dict
            df_metrics=pd.DataFrame(data=total_metrics)
            df_metrics.to_csv(os.path.join(logdir,'total_metrics.csv'))
            
        else: 
            assert self.task_name =='length_of_stay'
            all_preds=all_preds.squeeze(1).cpu()
            all_target=all_target.cpu()
            final_metrics=self.calculate_multiclass_for_one_class(all_preds, all_target)
            for k,v in final_metrics.items():
                self.log(f'test/{k}', v, prog_bar=True, on_epoch=True)
            with open(os.path.join(logdir,'total_metrics.json'), "w", encoding="utf-8") as f:
                json.dump(final_metrics, f, indent=4, ensure_ascii=False)  # ensure_ascii=False 支持中文

  
    def configure_optimizers(self):
            lr = self.learning_rate
            params = list(self.parameters())
            opt = torch.optim.AdamW(params, lr=lr)
            schedular=CosineAnnealingLR(opt,T_max=self.max_epochs,eta_min=1e-7)
            return [opt],[schedular]

    
    def calculate_binary_metrics(self, predictions, target):
        preds=predictions.cpu().numpy()
        target=target.cpu().numpy()
        # breakpoint()
        roc_auc = roc_auc_score(target, preds)
        pr_auc = average_precision_score(target, preds)

        return torch.tensor(roc_auc).to(self.device), torch.tensor(pr_auc).to(self.device)
    



    def calculate_multiclass_for_one_class(self, preds, target):    
        # Ensure inputs are numpy arrays
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Convert to probabilities
        preds = torch.softmax(torch.tensor(preds), dim=-1).numpy()
        
        # ROC AUC (One-vs-Rest)
        roc_auc = roc_auc_score(target, preds, multi_class='ovr')
        
        # PR AUC (Precision-Recall AUC) for each class then macro-average
        pr_auc_scores = []
        n_classes = preds.shape[1]
        for class_idx in range(n_classes):
            class_target = (target == class_idx).astype(int)  # Now safe as numpy
            precision, recall, _ = precision_recall_curve(
                class_target, 
                preds[:, class_idx]
            )
            pr_auc_scores.append(auc(recall, precision))
        pr_auc_macro = np.mean(pr_auc_scores)
        
        # Get predicted labels
        pred_labels = np.argmax(preds, axis=1)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(target, pred_labels)
        
        # Accuracy
        accuracy = accuracy_score(target, pred_labels)
        
        # Macro averages
        avg_precision_macro = precision_score(target, pred_labels, average='macro')
        avg_recall_macro = recall_score(target, pred_labels, average='macro')
        f1_macro = f1_score(target, pred_labels, average='macro')
        
        # Micro averages
        avg_precision_micro = precision_score(target, pred_labels, average='micro')
        avg_recall_micro = recall_score(target, pred_labels, average='micro')
        f1_micro = f1_score(target, pred_labels, average='micro')
        
        metrics = {
            'roc_auc': roc_auc,
            'pr_auc_macro': pr_auc_macro,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'kappa': kappa,
            'accuracy': accuracy,
            'avg_precision_micro': avg_precision_micro,
            'avg_recall_micro': avg_recall_micro,
            'avg_precision_macro': avg_precision_macro,
            'avg_recall_macro': avg_recall_macro,
        }
    
        return metrics

    
    def calculate_multiclass_metrics(self, predictions, target):
        # calculate precision and recall for evary class
        preds_ = predictions.cpu().numpy()
        target_ = target.cpu().numpy()
        total_metrics={}
        for cl in range(len(PROGRESSION_DISEASE)):#self.num_classes:
            preds=preds_[:,cl,:]
            target=target_[:,cl]
            # remove nan
            preds=preds[target!=3]
            target=target[target!=3]
            total_metrics[PROGRESSION_DISEASE[cl]]=self.calculate_multiclass_for_one_class(preds,target)
        return total_metrics

