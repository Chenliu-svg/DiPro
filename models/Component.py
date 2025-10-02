import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.nn import functional as F
from mimic_data_extract.mimic3benchmark.constants import PROGRESSION_DISEASE, FILL_VALUE
from einops import rearrange, repeat
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class OrderAwareProjection(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super().__init__()
        
        self.proj_net = nn.Sequential(
            nn.Linear(input_dim, 4*output_dim),
            nn.GELU(),
            nn.Linear(4*output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout_rate)
        )
        self.output_dim = output_dim
        
    def forward(self, pair_features):
        """
        input: pair_features(the anatomical features of neighboring CXRs): [B, P, R, 2D]
        output: forward_feat, reverse_feat: [B, P, R, D]
        """
        B, P, R, _ = pair_features.shape
        
        # generate reverse input by swapping the order of feature pairs
        reverse_features = torch.stack([
            pair_features[..., self.output_dim:],  # the second time point
            pair_features[..., :self.output_dim]   # the first time point
        ], dim=-1).view(B*P*R, 2*self.output_dim)
        
        flat_features = pair_features.view(-1, 2*self.output_dim)  # [B*P*R, 2D]
        reverse_flat = reverse_features.view(-1, 2*self.output_dim)
        
        forward_feat = self.proj_net(flat_features)
        reverse_feat = self.proj_net(reverse_flat)
  
        return (forward_feat.view(B, P, R, -1), 
                reverse_feat.view(B, P, R, -1))

class STD_Constraint(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0.1))
        
    def longitudinal_consistency_loss(self, static_features):
        # static_features: [B, P, R, D]
        anchor = static_features[:, :-1]  
        positive = static_features[:, 1:] 

        loss = F.mse_loss(anchor, positive, reduction='mean')
        return loss
    

    
    def orthogonal_projection_loss(self, static_features, dynamic_features):

        # static_features: [B, P, R, D], dynamic_features: [B, P, R, D]
        static_norm = F.normalize(static_features, p=2, dim=-1)
        dynamic_norm = F.normalize(dynamic_features, p=2, dim=-1)

        cosine_sim = torch.sum(static_norm * dynamic_norm, dim=-1)  # [B, P, R]
        loss = torch.mean(cosine_sim**2)
    
        return loss
    
    def forward(self, static_features, dynamic_features):
        B,P,R,D=static_features.shape
        orthogonal_loss = self.orthogonal_projection_loss(static_features, dynamic_features)
        consistency_loss=torch.tensor(0.0, device=static_features.device, requires_grad=True)
        if P>1:
            # consistency loss is only employed when there are at least 2 pairs
            consistency_loss = self.longitudinal_consistency_loss(static_features)
            
        return consistency_loss, orthogonal_loss

class LongitudinalCXRModel(nn.Module):
    def __init__(self, 
                 loss_ls,
                 device,
                 feature_size=256,
                 dropout_rate=0.3,
                 freeze_backbone=True,
                 ):
        super().__init__()
        self.feature_size=feature_size
        self.device=device
        # reginal feature extractor
        self.resnet = resnet50(pretrained=True)
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet.fc = nn.Identity()  
        self.feature_adapter = nn.Sequential(
            nn.Linear(2048, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # f_s in paper
        self.static_proj = OrderAwareProjection(2*feature_size, feature_size, dropout_rate)
        # f_d in paper
        self.dynamic_proj = OrderAwareProjection(2*feature_size, feature_size, dropout_rate)
        
        # the output layer for PAE module
        self.PAE_output_layer = nn.ModuleList()
        for _ in range(len(PROGRESSION_DISEASE)):
            self.PAE_output_layer.append(nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, 3) # the output is 3 classes (improve, no change, worsen)
        ))
            
        self.STD_Constraint=STD_Constraint()
        self.loss=loss_ls
        

    def forward(self, x, y_aux):
        """
        x: [B, T, R, C, H, W] (T: is the number of CXRs)
        y_aux: [B,T-1,R,num_progression_diseases]
        """
        B, T, R, C, H, W = x.shape
        P=T-1  # number of pairs of adjacent CXRs
        # ----------------- feature extraction -----------------
        #  [B*T*R, C, H, W]
        x_flat = x.view(-1, C, H, W)
        # [B*T*R, C, H, W]->[B*T*R, 2048]
        features = self.resnet(x_flat)
        
        # [B, T, R, feature_size]
        features = self.feature_adapter(features)
        features = features.view(B, T, R, -1)
        
        # ----------------- STD & PAE -----------------
        adjacent_pairs = []
        adjacent_pairs = torch.stack([
            torch.cat([features[:, t], features[:, t+1]], dim=-1)
            for t in range(P)
        ], dim=1)  # [B, P, R, 2D]

        # get static and reversed static features
        static, static_rev = self.static_proj(adjacent_pairs) # [B,P,R,D]

        # get dynamic and reversed dynamic features
        dynamic, dynamic_rev = self.dynamic_proj(adjacent_pairs)
        
        # In PAE, we encourage the consistency over the reversal of the static features
        static_loss = F.mse_loss(static, static_rev.detach())

        # In PAE, we leverage an exlpicit constraint on dynamic features using aux labels (disease progression labels of anatomical regions)
        num_progression_diseases = len(PROGRESSION_DISEASE)
        y_aux=y_aux.view(B*P*R,num_progression_diseases)
        y_aux_rev = torch.where((y_aux==0)|(y_aux==1), 
                             1-y_aux, 
                             y_aux) # [B*P*R,num_progression_diseases]
        y_real=torch.cat([y_aux,y_aux_rev],dim=0)

        output_logits=[]
        output_logits_rev=[]
        for cls_ in range(num_progression_diseases):
            output = self.PAE_output_layer[cls_](dynamic)
            output=output.view(B*P*R, 3)
            output_logits.append(output)
            output_rev = self.PAE_output_layer[cls_](dynamic_rev)
            output_rev=output_rev.view(B*P*R, 3)
            output_logits_rev.append(output_rev)
        output=torch.stack(output_logits,dim=1)
        output_rev=torch.stack(output_logits_rev,dim=1) #(B*P*R, num_progression_diseases, 3)

        final_output=torch.cat([output,output_rev],dim=0) #(2*B*P*R, num_progression_diseases, 3)
        dynamic_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for i in range(num_progression_diseases):
            disease_pred=final_output[:,i]
            gt=y_real[:,i].long()
            if torch.all((gt==FILL_VALUE)):
                continue
            disease_pred=disease_pred[gt!=FILL_VALUE]
            gt=gt[gt!=FILL_VALUE]
            t_fc=self.loss[i].to(disease_pred.device)
            dynamic_loss=dynamic_loss+t_fc(disease_pred,gt)

        # add longitudinal and orthogonal constraints in the STD module
        consistency_loss, orthogonal_loss = self.STD_Constraint(static, dynamic)
        
        
        return static, dynamic, consistency_loss, orthogonal_loss , static_loss, dynamic_loss

class LearnableRegionPadding(nn.Module):
    def __init__(self, region_shape):
        super().__init__()
        R, C, H, W = region_shape
        self.pad_value = nn.Parameter(torch.randn(R, C, H, W))
        
    def forward(self, x, mask):
        """
        x: [B, T, R, C, H, W] (T: is the number of CXRs)
        mask: [B, T]
        """
        B, T, R, C, H, W = x.shape
        
        # expand mask to the same shape as x [B, T, R, C, H, W]
        mask_expanded = mask.view(B, T, 1, 1, 1, 1).expand_as(x)
        
        # replace masked regions with learnable padding
        x_padded = torch.where(mask_expanded, x, self.pad_value)
        return x_padded


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EHRIntervalEncoder(nn.Module):
    """time interval aware ehr encoder"""
    def __init__(self, d_model, n_heads, sigma=1.0):
        super().__init__()
        self.num_heads=n_heads
        self.sigma = nn.Parameter(torch.tensor(sigma))  
        self.time_encoder = nn.Sequential(
            nn.Linear(3, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads)
    
    def get_time_weights(self, ehr_t, start, end, mode='end', sigma=1.0):
        """
        mode: 'end', 'edge', 'center'
        """
    
        inf_tensor = torch.tensor(float('inf'), dtype=ehr_t.dtype, device=ehr_t.device)

       
        if mode == 'end':
            time_diff = torch.where((ehr_t >= start) & (ehr_t <= end), end - ehr_t, inf_tensor)
        elif mode == 'edge':
            
            time_diff = torch.where((ehr_t >= start) & (ehr_t <= end), 
                                torch.min(ehr_t - start, end - ehr_t), inf_tensor)
        elif mode == 'center':
            center = (start + end) / 2
            time_diff = torch.where((ehr_t >= start) & (ehr_t <= end), 
                                torch.abs(ehr_t - center), inf_tensor)
        else:
            raise ValueError("Mode must be 'end', 'edge', or 'center'")
        
        
        return -time_diff

    def forward(self, ehr, ehr_times, interval_times):
        """
        ehr: (B, L, D) 
        ehr_times: (B, L) 
        interval_times: (B, P, 2)  [start_time, end_time]
        """
        B, L, D = ehr.shape
        P = interval_times.shape[1]
        
        ehr_t = repeat(ehr_times, 'b l -> b t l', t=P)
        start = repeat(interval_times[..., 0], 'b t -> b t l', l=L)
        end = repeat(interval_times[..., 1], 'b t -> b t l', l=L)
        
        # construct time features
        time_features = torch.stack([
            ehr_t - start,
            end - ehr_t,
            torch.sigmoid((ehr_t - start)*(end - ehr_t))  #  whether in the interval
        ], dim=-1)  # (B, P, L, 3)
        
        # generate time embeddings
        time_emb = self.time_encoder(time_features)  # (B, P, L, D)
        ehr_expanded = repeat(ehr, 'b l d -> b t l d', t=P)
        
        # original attn_mask: [B, P, L]
        attn_mask=self.get_time_weights(ehr_t, start, end, mode='center', sigma=self.sigma)
        attn_mask = attn_mask.unsqueeze(-2)  # [B, P, 1, L]
        attn_mask = attn_mask.expand(-1, -1, L, -1)  #  -> [B, P, L, L]
        attn_mask=attn_mask.view(B*P, L,L)
        attn_mask=repeat(attn_mask,'b l d -> (b n) l d', n=self.num_heads)
        
        attn_output, _ = self.attn(
            rearrange(time_emb, 'b t l d -> l (b t) d'),  # using time interval as query
            rearrange(ehr_expanded, 'b t l d -> l (b t) d'),
            rearrange(ehr_expanded, 'b t l d -> l (b t) d'),
            attn_mask=attn_mask
        )
        return rearrange(attn_output, 'l (b t) d -> b t l d', b=B)  # (B, P, L, D)


class EHRTransformer_cls(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=8,
        num_layers=1,
        dropout=0.1
    ):
        super().__init__()
        
        total_dim = input_dim
        self.cls_token=nn.Parameter(torch.randn(1,d_model))

        self.input_proj = nn.Sequential(
            nn.Linear(total_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4*d_model,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
       
    def forward(self, x, mask):
        """
        x: [batch_size, max_timesteps, num_features]
        mask: [batch_size, max_timesteps] (1 for valid, 0 for padding)
        """
        proj = self.input_proj(x)
        
        # Positional Encoding
        proj = self.pos_encoder(proj)
        
        # add cls token
        cls_tokens = self.cls_token.expand(proj.shape[0],-1,-1)
        proj = torch.cat((cls_tokens,proj),dim=1)

        # generate Transformer padding mask
        padding_mask = ~mask.bool()  
        # add padding mask for cls token
        padding_mask = torch.cat((torch.zeros(proj.shape[0],1).bool().to(padding_mask.device),padding_mask),dim=1)

        # Transformer处理
        memory = self.transformer(
            src=proj,
            src_key_padding_mask=padding_mask
        )
        
        # cls token
        cls=memory[:,0]
        memory=memory[:,1:]
    
        return cls,memory


class EHRCrossAttention(nn.Module):
    def __init__(self,d_model=256, num_heads=8):
        super().__init__()
        
        self.stage1_attn_CXR_EHR = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1_CXR_EHR = nn.LayerNorm(d_model)

    def forward(self, cxr, cxr_mask, ehr):
        """
        input:
        cxr: [batch, num_cxr, D]
        ehr: [batch, num_ehr, D] 
        demo: [batch, num_demo, D]
        output:
        [batch, D]
        """
    
        attn_out1_cxr_ehr, _ = self.stage1_attn_CXR_EHR(
            query=ehr,  # [B, num_cxr, D]
            key=cxr,        # [B, num_ehr, D]
            value=cxr   # [B, num_ehr, D]
            
        )
        ehr_enhanced = self.norm1_CXR_EHR(ehr + attn_out1_cxr_ehr)  # [B, cxr_len, D]

        
        return ehr_enhanced # [B,ehr_len, D]

class RegionAwareAttention(nn.Module):
    def __init__(self, d_model, n_heads):        
        super().__init__()
        self.region_proj = nn.Linear(d_model, d_model*3)
        self.ehr_proj = nn.Linear(d_model, d_model*2)
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
        
        
    def forward(self, regions, ehr_feats=None):
        """
        regions: (B*T, R, D)
        ehr_feats: # (B, T, L, D)
        """
        ehr_feats = repeat(ehr_feats, 'b t l d -> (b t) l d')
        
        region_q, region_k, region_v = self.region_proj(regions).chunk(3, dim=-1)
        ehr_k, ehr_v = self.ehr_proj(ehr_feats).chunk(2, dim=-1)
        
        q = region_q
        k = torch.cat([region_k, ehr_k], dim=1)
        v = torch.cat([region_v, ehr_v], dim=1)
        
        attn_output, _ = self.attn(q.transpose(0,1), k.transpose(0,1), v.transpose(0,1))
        return self.norm(attn_output.transpose(0,1) + regions)



class MultisacleMultimodalFusion(nn.Module):
    def __init__(self, num_final_class, ehr_num_var, d_model=256, n_heads=8, num_layers=4, num_fusion_layers=1, dropout_rate=0.1):
        super().__init__()
        self.num_final_class=num_final_class
        self.cls_token=nn.Parameter(torch.randn(1, num_final_class, d_model))
        self.ehr_global_encoder=EHRTransformer_cls(d_model=d_model,input_dim=ehr_num_var)
        self.ehr_local_encoder = EHRIntervalEncoder(d_model, n_heads)

        self.region_attn = nn.ModuleList([
            RegionAwareAttention(d_model, n_heads) 
            for _ in range(num_layers//2)
        ])
        
        self.ehr_cross_attn=EHRCrossAttention(d_model=d_model, num_heads=n_heads)
        
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model,
                nhead=8,
                dim_feedforward=4*d_model,
                dropout=dropout_rate,
                batch_first=True
            ),
            num_layers=num_fusion_layers
        )

        
    def forward(self, cxr, reshaped_region_mask, ehr, ehr_mask, ehr_times, interval_times):
        """
        cxr: (B, P, R, D)
        reshaped_region_mask: (B, P)
        ehr: (B, L, D)
        ehr_times: (B, L) 
        interval_times: (B, P, 2) 
        """
        B, P, R, D = cxr.shape
        cls_token=self.cls_token.expand(B, -1, -1)
        device = reshaped_region_mask.device
        
        # global-level ehr encoding 
        cls, ehr = self.ehr_global_encoder(ehr, ehr_mask)

        # Local CXR-EHR fusion within temporal intervals
        ehr_feats = self.ehr_local_encoder(ehr, ehr_times, interval_times)  # (B, T, D)
        regions = rearrange(cxr, 'b t r d -> (b t) r d')
        for layer in self.region_attn:
            regions = layer(regions, ehr_feats)
        
        #  refine the global EHR representation by attending over all local CXR regions
        temporal_feats = rearrange(regions, '(b t) r d -> (b r) t d', b=B)
        temporal_feats = rearrange(temporal_feats, '(b r) t d -> b (t r) d', b=B, t=P)
        reshaped_region_mask=repeat(reshaped_region_mask, 'b t -> b (t r)', r=R)
        ehr_enhanced = self.ehr_cross_attn(temporal_feats,  reshaped_region_mask, ehr)

        #  include an additional self-attention layer to further enhance global interactions
        total_mask = torch.cat([
            torch.ones((B, self.num_final_class), dtype=torch.bool, device=device),
            reshaped_region_mask,
            torch.ones(ehr.shape[:2], dtype=torch.bool, device=device)
        ], dim=1)
        concat_ehr_cxr=torch.cat([cls_token, temporal_feats,ehr_enhanced], dim=1)
        fused_ehr_cxr=self.fusion(concat_ehr_cxr, src_key_padding_mask=~total_mask)
        
        return fused_ehr_cxr  # (B, L', D)


class GroupEmbedding(nn.Module):
    """ 
    GE(X) = softmax(XW)G
    
    X: [batch, num_vars, D]
    W: [D, K] 
    G: [K, D]
    K < D
    """
    def __init__(self, D: int, K: int):
        super().__init__()
        assert K < D, "K must be smaller than D"
        
        # Initialize
        self.W = nn.Parameter(torch.Tensor(D, K))  
        self.G = nn.Parameter(torch.Tensor(K, D))  
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.G)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [batch_size, num_variables, D]
        output: [batch_size, num_variables, D] 
        """
        batch, num_vars, D = X.shape
    
        assert D == self.W.size(0), "the input feature dimension must match W"
        
        # Step 1: XW → [batch, num_vars, K]
        XW = torch.matmul(X, self.W)  
        
        # Step 2: softmax(XW)
        alpha = F.softmax(XW, dim=-1)  # [batch, num_vars, K]
        
        # Step 3: alphaG → [batch, num_vars, D]
        output = torch.einsum('bnk,kd->bnd', alpha, self.G)
        
        return output

