import random

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN
from timm.models.layers import DropPath, trunc_normal_

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from peft.adapter import LinearAdapter
from segmentation.pointnet_util import index_points, square_distance
from utils import misc
from utils.checkpoint import (get_missing_parameters_message,
                              get_unexpected_parameters_message)
from utils.logger import *

from .build import MODELS


class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


# TODO: Benchmark Runtime
class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, return_idx=False):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
            idx : B G M
            center_idx : B G
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center, center_idx = misc.fps(xyz, self.num_group, return_idx=True) # B G 3, B G
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        if return_idx:
            return neighborhood, center, idx.view(batch_size, self.num_group, self.group_size), center_idx
        else:
            return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, 
        dim,
        bottleneck_dim=None,
        num_heads=8, 
        qkv_bias=False, 
        qk_scale=None, 
        attn_drop=0., 
        proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.bottleneck_dim = bottleneck_dim
        
        if bottleneck_dim is None:
            # Standard attention
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
        else:
            # Bottlenecked attention
            self.scale = qk_scale or bottleneck_dim ** -0.5
            self.qkv = nn.Linear(dim, bottleneck_dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(bottleneck_dim, dim)
            
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        
        if self.bottleneck_dim is None:
            # Standard attention path
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            # Bottlenecked attention path
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.bottleneck_dim // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, self.bottleneck_dim)
            
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        drop=0., 
        attn_drop=0.,
        drop_path=0., 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm,
        peft=False,
        bottleneck_dim=None,
        prompt_prior_adapter_drop_rate=0.0,
        geometric_adapter_drop_rate=0.0,
        output_adapter_drop_rate=0.0,
        num_tokens_k=10,
        max_peft_depth=5,
        geometric_adapter_scale=0.7,
        pooling_scale=0.3
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # PEFT
        if peft:
            self.prompt_prior_adapter = LinearAdapter(
                dim,
                bottleneck_dim=bottleneck_dim,
                drop_rate=prompt_prior_adapter_drop_rate,
                return_residual=True
            )
            
            self.geometric_adapter = LinearAdapter(
                dim,
                bottleneck_dim=bottleneck_dim,
                drop_rate=geometric_adapter_drop_rate,
                return_residual=False
            )
            
            self.output_adapter = LinearAdapter(
                dim,
                bottleneck_dim=bottleneck_dim,
                drop_rate=output_adapter_drop_rate,
                return_residual=True
            )
            
            self.output_transform = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.GELU()
            )
            
            self.prompt_dropout = nn.Dropout(0.1)
            self.K = num_tokens_k
            self.max_peft_depth = max_peft_depth
            self.geometric_adapter_scale = geometric_adapter_scale
            self.pooling_scale = pooling_scale
            self.prompt_embedding_parameters = nn.Parameter(torch.zeros(self.K, dim))
            trunc_normal_(self.prompt_embedding_parameters, std=.02)
            
    def pooling(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(dim=2)[0]
        lc_x = self.out_transform(lc_x.permute(0, 2, 1)).permute(0,2,1)
        return lc_x
    
    def propagate(self, xyz1, xyz2, points1, points2, de_neighbors):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, N, D']
            points2: input points data, [B, S, D'']
        Return:
            new_points: upsampled points data, [B, N, D''']
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :de_neighbors], idx[:, :, :de_neighbors]  # [B, N, S]
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        weight = weight.view(B, N, de_neighbors, 1)
        interpolated_points = torch.sum(index_points(points2, idx) * weight, dim=2)#B, N, 6, C->B,N,C
        new_points = points1+0.3*interpolated_points # B,N,C
        return new_points
        
    def forward(
        self, 
        x,
        mask=None,
        center=None,
        second_center=None,
        second_idx=None,
        second_center_idx=None,
        second_group_size=None,
        prompt_prior=None,
        local_attention=None,
        local_attention_norm=None,
        layer_id=None
    ):
        
        B, G, _ = x.shape
        if mask is not None:
            _, mask_dim, _ = mask.shape
            mask_new = torch.zeros([B, mask_dim+self.K+1, mask_dim+self.K+1]).cuda()
            mask_new[:, self.K+1:, self.K+1:] = mask
            mask = mask_new
            
        if layer_id < self.max_peft_depth:
            prompt = self.prompt_dropout(self.prompt_embeddings.repeat(B, 1, 1))
            
            if prompt_prior is not None:
                adapted_prompt = self.prompt_prior_adapter(prompt_prior)
                prompt = prompt + adapted_prompt
                
            x = torch.cat((x[:,0].unsqueeze(1), prompt, x[:,1:]), 1)
            x = x + self.attn(self.norm1(x), prompt, mask)[0]
            x_fn = self.drop_path(self.mlp(self.norm2(x)))
            x = x + x_fn + self.geometric_adapter_scale * self.geometric_adapter(x_fn)
            
            prompt = x[:, 1:self.K+1]
            x = torch.cat((x[:,0].unsqueeze(1), x[:, self.K+1:]), 1)
            cls_x = x[:, 0]
            x = x[:, 1:]
            
            prompt_x = torch.cat((prompt, x), dim=1)
            
            x_neighborhoods = prompt_x.reshape(B*G, -1)[second_idx, :]
            x_neighborhoods = x_neighborhoods.reshape(B*second_center.shape[1], second_group_size, -1)
            x_centers = prompt_x.reshape(B*G, -1)[second_center_idx, :]
            x_centers = x_centers.reshape(B, second_center.shape[1], -1)
            
            x_neighborhoods = x_neighborhoods.clone() + self.drop_path(local_attention(local_attention_norm(x_neighborhoods.clone())))
            
            vis_x = self.pooling(x_neighborhoods.reshape(B, second_center.shape[1], second_group_size, -1))
            vis_x = vis_x + self.pooling_scale * x_centers
            
            # TODO: Replace algorithmic propagation with learned propagation layer
            # Current propagation uses fixed geometric rules
            # Could use a learned MLP or attention mechanism to propagate features
            # between points in a data-driven way
            x = self.propagate(xyz1=center, xyz2=second_center, points1=x, points2=vis_x, 
                            de_neighbors=second_center.shape[1])
            
            x = torch.cat((cls_x.unsqueeze(1), prompt, x), 1)
            x = self.output_adapter(x)
            
            # Remove prompt tokens
            x = torch.cat((x[:,0].unsqueeze(1), x[:, self.K+1:]), 1)
        else:
            x = x + self.attn(self.norm1(x), mask)[0]
            x_fn = self.drop_path(self.mlp(self.norm2(x)))
            x = x + x_fn + self.geometric_adapter_scale * self.geometric_adapter(x_fn)
            cls_x = x[:,0]
            x = x[:,1:]
            G = G-1
            
            x_neighborhoods = x.reshape(B*G, -1)[second_idx, :].reshape(B*second_center.shape[1], second_group_size, -1)
            x_centers = x.reshape(B*G, -1)[second_center_idx, :].reshape(B, second_center.shape[1], -1)
            
            x_neighborhoods = x_neighborhoods.clone() + self.drop_path(local_attention(local_attention_norm(x_neighborhoods.clone())))
            
            vis_x = self.pooling(x_neighborhoods.reshape(B, second_center.shape[1], second_group_size, -1)) + self.pooling_scale * x_centers
            x = self.propagate(xyz1=center, xyz2=second_center, points1=x, points2=vis_x, de_neighbors=second_center.shape[1])
            
            x = torch.cat((cls_x.unsqueeze(1), x), 1)
            x = self.output_adapter(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., peft=False, bottleneck_dim=None, 
                 prompt_prior_adapter_drop_rate=0.0, geometric_adapter_drop_rate=0.0, output_adapter_drop_rate=0.0, 
                 num_tokens_k=10, max_peft_depth=5, geometric_adapter_scale=0.7, pooling_scale=0.3):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                peft=peft,
                bottleneck_dim=bottleneck_dim,
                prompt_prior_adapter_drop_rate=prompt_prior_adapter_drop_rate,
                geometric_adapter_drop_rate=geometric_adapter_drop_rate,
                output_adapter_drop_rate=output_adapter_drop_rate,
                num_tokens_k=num_tokens_k,
                max_peft_depth=max_peft_depth,
                geometric_adapter_scale=geometric_adapter_scale,
                pooling_scale=pooling_scale
                )
            for i in range(depth)])
        
        self.local_attention = Attention(
            dim=embed_dim,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate
        )
        
        self.local_attention_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        x, 
        pos,
        mask=None,
        center=None,
        second_center=None,
        second_idx=None,
        second_center_idx=None,
        second_group_size=None,
        prompt_prior=None,
        layer_id=None
    ):
        for _, block in enumerate(self.blocks):
            x = block(
                x + pos,
                mask=mask,
                center=center,
                second_center=second_center,
                second_idx=second_idx,
                second_center_idx=second_center_idx,
                second_group_size=second_group_size,
                prompt_prior=prompt_prior,
                local_attention=self.local_attention,
                local_attention_norm=self.local_attention_norm,
                layer_id=layer_id
            )
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug = False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


@MODELS.register_module()
class Point_MAE_PEFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE_PEFT] ', logger='Point_MAE_PEFT')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE_PEFT] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE_PEFT')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()
            
    def encode_pts(self, pts):
        """
        Encode the point cloud into a set of tokens.
        
        Args:
            pts: B N 3
        Returns:
            x_vis: B C
        """
        neighborhood, center = self.group_divider(pts)
        x_vis, _ = self.MAE_encoder(neighborhood, center)
        # perform max pooling over the group size so that the output is B C
        # TODO: Benchmark different pooling methods. Current one might be limiting for physics modeling.
        x_vis = torch.max(x_vis, dim=2)[0]
        return x_vis
    
    def forward(self, pts, vis = False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B,_,C = x_vis.shape # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _,N,_ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B*M,-1,3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        if vis: #visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1

# finetune model
@MODELS.register_module()
class PointTransformer_PEFT(nn.Module):
    def __init__(
        self, 
        config, 
        **kwargs
    ):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        
        # TODO: Move PEFT hyperparameters to config
        # Should define in config:
        # - bottleneck_dim: Dimension of bottleneck adapters
        # - prompt_prior_adapter_drop_rate: Dropout rate for prompt prior adapter
        # - geometric_adapter_drop_rate: Dropout rate for geometric adapter
        # - output_adapter_drop_rate: Dropout rate for output adapter
        # - num_tokens_k: Number of prompt tokens
        # - max_peft_depth: Maximum depth for PEFT layers
        # - geometric_adapter_scale: Scale factor for geometric adapter
        # - pooling_scale: Scale factor for pooling
        self.bottleneck_dim = 64
        self.prompt_prior_adapter_drop_rate = 0.1
        self.geometric_adapter_drop_rate = 0.1
        self.output_adapter_drop_rate = 0.1
        self.num_tokens_k = 10
        self.max_peft_depth = 5
        self.geometric_adapter_scale = 0.7
        self.pooling_scale = 0.3
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
            peft=True,
            bottleneck_dim=self.bottleneck_dim,
            prompt_prior_adapter_drop_rate=self.prompt_prior_adapter_drop_rate,
            geometric_adapter_drop_rate=self.geometric_adapter_drop_rate,
            output_adapter_drop_rate=self.output_adapter_drop_rate,
            num_tokens_k=self.num_tokens_k,
            max_peft_depth=self.max_peft_depth,
            geometric_adapter_scale=self.geometric_adapter_scale,
            pooling_scale=self.pooling_scale
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.finetune_head = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 8192)
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        
        # PEFT
        assert "k" in kwargs, "k must be provided"
        self.k = kwargs["k"]
        
        assert "prompt_bank" in kwargs, "prompt_bank must be provided"
        self.prompt_bank = kwargs["prompt_bank"]
        
        assert "prompt_encoder" in kwargs, "prompt_encoder must be provided"
        self.prompt_encoder = kwargs["prompt_encoder"]
        
        self.second_group_divider = Group(
            num_group=self.num_group // 2, group_size=self.group_size // 2)
        
        assert "embedding_model" in kwargs, "embedding_model must be provided"
        self.embedding_model = kwargs["embedding_model"]
        
    def prepare_for_peft(self):
        self = prepare_for_peft(self)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def build_prompt_prior(
        self,
        pts
    ):
        # Encode input points into features
        features = self.embedding_model.encode_pts(pts)
        
        # Query Chroma DB to find top K similar prompts
        results = self.prompt_bank.query(
            query_embeddings=features.detach().cpu().numpy(),
            n_results=self.k-2, # as implemented in the paper
            include=["embeddings", "distances"]
        )
        
        # Get the embeddings from results
        prompt_embeddings = torch.tensor(results['embeddings']).to(pts.device)  # Shape: [K-2, embedding_dim]
        similarity_scores = torch.tensor(results['distances']).unsqueeze(0).to(pts.device)  # Shape: [1, K-2]
        
        # Matrix multiply similarity scores with prompt embeddings
        # similarity_scores: [1, K-2], prompt_embeddings: [K-2, embedding_dim]
        # Output shape: [1, embedding_dim]
        weighted_prompts = torch.matmul(similarity_scores, prompt_embeddings)
        
        # Expand features to match batch dimension
        features = features.unsqueeze(1)  # Shape: [B, 1, embedding_dim]
        weighted_prompts = weighted_prompts.unsqueeze(1)  # Shape: [B, 1, embedding_dim]
        prompt_embeddings = prompt_embeddings.unsqueeze(0).expand(features.size(0), -1, -1)  # Shape: [B, K-2, embedding_dim]
        
        # Concatenate along K dimension
        # features: [B, 1, embedding_dim]
        # weighted_prompts: [B, 1, embedding_dim] 
        # prompt_embeddings: [B, K-2, embedding_dim]
        # Final shape: [B, K, embedding_dim] 
        prompt_features = torch.cat([features, weighted_prompts, prompt_embeddings], dim=1)
        return prompt_features

    def forward(
        self, 
        pts
    ):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        
        # PEFT
        prompt_prior = self.build_prompt_prior(pts)
        # second hierarchy
        _, \
            second_center, \
            second_idx, \
            second_center_idx = self.second_group_divider(pts, return_idx=True)
        ###
        
        # transformer
        x = self.blocks(
            x, 
            pos,
            mask=None,
            center=center,
            second_center=second_center,
            second_idx=second_idx,
            second_center_idx=second_center_idx,
            prompt_prior=prompt_prior,
            layer_id=None
        )
        x = self.norm(x)
        
        # TODO: Benchmark different pooling methods. Current one seems limiting for physics modeling.
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        # TODO: Benchmark neural operators for better physics performance.
        ret = self.finetune_head(concat_f)
        return ret

def prepare_for_peft(
    model: PointTransformer_PEFT
):
    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # unfreeze the prompt prior
    model.finetune_head.requires_grad = True
    
    # unfreeze the prompt prior
    model.blocks.local_attention.requires_grad = True
    model.blocks.local_attention_norm.requires_grad = True
    
    for block in model.blocks.blocks:
        block.prompt_prior.requires_grad = True
        block.geometric_adapter.requires_grad = True
        block.output_adapter.requires_grad = True
        block.output_transform.requires_grad = True
        block.prompt_embedding_parameters.requires_grad = True
    
    return model
    