
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from models.attention import Attention
from models.initialize import init_weight, init_activation


class BaseMLP(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 hidden_dims=[64] * 3, activation="relu", weight_initialization="xavier_uniform", bias=True):
        super(BaseMLP, self).__init__()

        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims + [output_dim]):
            layer = nn.Linear(in_dim, out_dim, bias=bias)
            init_weight(layer.weight, weight_initialization)
            self.layers.append(layer)

        self.activation = init_activation(activation)  

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x


class DeterministicEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, h_dim, num_layers=3, attn_type="uniform", self_attn=False):
        super(DeterministicEncoder, self).__init__()

        self.model = BaseMLP(x_dim + y_dim, r_dim, hidden_dims=[h_dim]*num_layers)
        self.self_attn = self_attn
        if self.self_attn:
            self.self_attention = Attention(r_dim, r_dim, attn_type=attn_type)
        self.cross_attention = Attention(x_dim, r_dim, attn_type=attn_type)

    def forward(self, x_context, y_context, x_target):
        """
            x_context: (batch_size * num_context * x_dim)
            y_context: (batch_size * num_context * y_dim)
            x_target: (batch_size * num_target * x_dim)
            self_attn: Boolean determining whether to apply self-attention

            output: (batch_size * num_target * r_dim)
        """
        xy_context = torch.cat([x_context, y_context], dim=-1)
        r = self.model(xy_context)
        if self.self_attn:
            r = self.self_attention(r, r, r)
        return self.cross_attention(x_target, x_context, r)


class LatentEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, h_dim, num_layers=3, attn_type="uniform", self_attn=False):
        super(LatentEncoder, self).__init__()

        self.model = BaseMLP(x_dim + y_dim, h_dim, hidden_dims=[h_dim]*(num_layers-1))
        self.self_attn = self_attn
        if self.self_attn:
            self.self_attention = Attention(h_dim, h_dim, attn_type=attn_type)

        self.mu_head = nn.Linear(h_dim, z_dim)
        init_weight(self.mu_head.weight, initialization="xavier_uniform")
        
        self.sigma_head = nn.Linear(h_dim, z_dim)
        init_weight(self.sigma_head.weight, initialization="xavier_uniform")

    def forward(self, x, y):
        """
            x: (batch_size * num_samples * x_dim)
            y: (batch_size * num_samples * y_dim)
            self_attn: Boolean determining whether to apply self-attention

            output: Normal(batch_size * z_dim)
        """
        x = torch.cat([x, y], dim=-1)
        h = self.model(x)
        if self.self_attn:
            h = self.self_attention(h, h, h)

        mu = self.mu_head(h).mean(dim=1)
        sigma = 0.1 + 0.9 * torch.sigmoid(self.sigma_head(h).mean(dim=1))
        return Normal(mu, sigma)


class Decoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim, num_layers=3):
        super(Decoder, self).__init__()

        self.model = BaseMLP(x_dim + r_dim + z_dim, h_dim, hidden_dims=[h_dim]*(num_layers-1))
        self.mu_head = nn.Linear(h_dim, y_dim)
        self.sigma_head = nn.Linear(h_dim, y_dim)

    def forward(self, x, r, z):
        """
            x: (batch_size * num_samples * x_dim)
            r: (batch_size * num_samples * r_dim)
            z: (batch_size * num_samples * z_dim)

            output: (batch_size * nun_samples * y_dim)
        """
        x = torch.cat([x, r, z], dim=-1)
        h = self.model(x)

        mu = self.mu_head(h)
        sigma = 0.1 + 0.9 * F.softplus(self.sigma_head(h))
        return Normal(mu, sigma)    


        