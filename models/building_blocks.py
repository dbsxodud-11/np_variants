
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from models.attention import MultiheadAttention, SelfAttention


class BaseMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(BaseMLP, self).__init__()

        hidden_dims = [hidden_dim] * (num_layers-1)
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims + [output_dim]):
            layer = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(layer.weight)
            self.layers.append(layer)

        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        return self.layers[-1](x)


class DeterministicEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, h_dim, pre_num_layers=4, post_num_layers=2, self_attn=False):
        super(DeterministicEncoder, self).__init__()

        if self_attn:
            self.pre_model = nn.Sequential(BaseMLP(x_dim + y_dim, h_dim, h_dim, num_layers=pre_num_layers-2),
                                           nn.ReLU(),
                                           SelfAttention(h_dim, h_dim))
        else:
            self.pre_model = BaseMLP(x_dim + y_dim, h_dim, h_dim, num_layers=pre_num_layers)
        
        self.post_model = BaseMLP(h_dim, r_dim, h_dim, num_layers=post_num_layers)

    def forward(self, x, y, mask=None):
        h = self.pre_model(torch.cat([x, y], dim=-1))
        if mask is None:
            h = h.mean(dim=-2)
        else:
            mask = mask.to(dtype=h.dtype, device=h.device)
            out = (out * mask.unsqueeze(-1)).sum(dim=-2) / (mask.sum(dim=-1, keepdim=True).detach() + 1e-5)
        
        return self.post_model(h)


class LatentEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, h_dim, pre_num_layers=3, post_num_layers=2, self_attn=False):
        super(LatentEncoder, self).__init__()

        if self_attn:
            self.pre_model = nn.Sequential(BaseMLP(x_dim + y_dim, h_dim, h_dim, num_layers=pre_num_layers-2),
                                           nn.ReLU(),
                                           SelfAttention(h_dim, h_dim))
        else:
            self.pre_model = BaseMLP(x_dim + y_dim, h_dim, h_dim, num_layers=pre_num_layers)

        self.post_model = BaseMLP(h_dim, z_dim*2, h_dim, num_layers=post_num_layers)

    def forward(self, x, y, mask=None):
        h = self.pre_model(torch.cat([x, y], dim=-1))
        if mask is None:
            h = h.mean(dim=-2)
        else:
            mask = mask.to(dtype=h.dtype, device=h.device)
            out = (out * mask.unsqueeze(-1)).sum(dim=-2) / (mask.sum(dim=-1, keepdim=True).detach() + 1e-5)
        
        mu, pre_sigma = self.post_model(h).chunk(2, dim=-1)
        sigma = 0.1 + 0.9 * torch.sigmoid(pre_sigma)

        return Normal(mu, sigma)


class CrossAttentionEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, h_dim, qk_num_layers=2, v_num_layers=4, self_attn=True):
        super(CrossAttentionEncoder, self).__init__()

        self.qk_pre_model = BaseMLP(x_dim, h_dim, h_dim, num_layers=qk_num_layers)
        self.self_attn = self_attn

        if self.self_attn:
            self.v_pre_model = BaseMLP(x_dim + y_dim, h_dim, h_dim, num_layers=v_num_layers-2)
            self.self_attention = SelfAttention(h_dim, h_dim)
        else:
            self.v_pre_model = BaseMLP(x_dim + y_dim, h_dim, h_dim, num_layers=v_num_layers)

        self.cross_attention = MultiheadAttention(h_dim, h_dim, h_dim, h_dim)

    def forward(self, x_context, y_context, x_target, mask=None):
        q, k = self.qk_pre_model(x_target), self.qk_pre_model(x_context)
        v = self.v_pre_model(torch.cat([x_context, y_context], dim=-1))

        if self.self_attn:
            v = self.self_attention(v, mask=mask)

        return self.cross_attention(q, k, v, mask=mask)


class Decoder(nn.Module):
    def __init__(self, x_dim, y_dim, enc_dim, h_dim, num_layers=3, bootstrap=False):
        super(Decoder, self).__init__()

        self.bootstrap = bootstrap

        if self.bootstrap:
            self.fc_base = nn.Linear(x_dim + enc_dim, h_dim)
            self.fc_bootstrap = nn.Linear(enc_dim, h_dim)
            self.model = BaseMLP(h_dim, y_dim * 2, h_dim, num_layers=num_layers-1)
        else:
            self.model = BaseMLP(x_dim + enc_dim, y_dim * 2, h_dim, num_layers=num_layers)     

    def forward(self, x, enc, extra_enc=None):
        if self.bootstrap:
            h_base = self.fc_base(torch.cat([x, enc], dim=-1))
            h_bootstrap = self.fc_bootstrap(torch.cat([extra_enc]))
            h = F.relu(h_base + h_bootstrap)
        else:
            h = torch.cat([x, enc], dim=-1)
        
        mu, pre_sigma = self.model(h).chunk(2, dim=-1)
        sigma = 0.1 + 0.9 * torch.sigmoid(pre_sigma)

        return Normal(mu, sigma)   
