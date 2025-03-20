import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LinearAdapter(nn.Module):
    def __init__(self,
        embed_dim,
        bottleneck_dim,
        drop_rate=0,
        return_residual=True
    ):
        super(LinearAdapter, self).__init__()
    
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim

        self.dropout = nn.Dropout(p=drop_rate)
        self.identity = False
        self.return_residual = return_residual

        if self.bottleneck_dim > 0:
            self.ln1 = nn.Linear(self.embed_dim, self.bottleneck_dim)
            self.activate = QuickGELU()
            self.ln2 = nn.Linear(self.bottleneck_dim, self.embed_dim)
            self.init_weights()
        
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init_weights)

    def set_sample_config(self, sample_embed_dim):
        self.identity = False
        self.sample_embed_dim = sample_embed_dim
        if self.sample_embed_dim == 0:
            self.identity = True
        else:
            self.sampled_weight_0 = self.ln1.weight[:self.sample_embed_dim,:]
            self.sampled_bias_0 =  self.ln1.bias[:self.sample_embed_dim]
            self.sampled_weight_1 = self.ln2.weight[:, :self.sample_embed_dim]
            self.sampled_bias_1 =  self.ln2.bias


    def forward(self, x, identity=None):
        if self.identity:
            return x
        out = self.ln1(x)
        out = self.activate(out)
        out = self.dropout(out)
        out = self.ln2(out)
        if self.return_residual:
            if identity is None:
                identity = x
            return identity + out
        else:
            return out

    def calc_sampled_param_num(self):
        if self.identity:
            return 0
        else:
            return  self.sampled_weight_0.numel() + \
                self.sampled_bias_0.numel() + \
                self.sampled_weight_1.numel() + \
                self.sampled_bias_1.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias_0 is not None:
             total_flops += self.sampled_bias_0.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight_0.size())
        return total_flops