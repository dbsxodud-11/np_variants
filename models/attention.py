import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_attention(q, v):
    """
        Uniform attention. Equivalent to mean.
        q: (batch_size * num_queries * q_dim)
        v: (batch_size * num_values * v_dim)

        output: (batch_size * v_dim)
    """
    _, num_queries, _ = q.size()
    return v.mean(dim=1).unsqueeze(1).repeat(1, num_queries, 1)

def laplace_attention(q, k, v, scale=1.0, normalize=False):
    """
        Laplace Exponential attention.
        q: (batch_size * num_queries * q_dim)
        k: (batch_size * num_keys * k_dim)
        v: (batch_size * num_values * v_dim)

        scale: float that scales the L1 distance.
        normalize: Boolean that determines whether weights sum to 1.

        output: (batch-size * num_queries * v_dim)
    """
    batch_size, num_queries, x_dim = q.size()
    _, num_keys, _ = v.size()

    q = q.view(batch_size, num_queries, 1, x_dim)
    k = k.view(batch_size, 1, num_keys, x_dim)

    weights = torch.abs((k - q) / scale).sum(axis=-1)
    if normalize:
        weights = F.softmax(weights, dim=-1)
    else:
        weights = 1.0 + torch.tanh(weights)
    return torch.matmul(weights, v)

def dot_product_attention(q, k, v, scale=1.0, normalize=False):
    """
        Dot Product attention.
        q: (batch_size * num_queries * q_dim)
        k: (batch_size * num_keys * k_dim)
        v: (batch_size * num_values * v_dim)
        normalize: Boolean that determines whether weights sum to 1.

        output: (batch-size * num_queries * v_dim)
    """
    scale = math.sqrt(q.shape[-1])
    weights = torch.matmul(q, k.transpose(1, 2)) / scale
    if normalize:
        weights = F.softmax(weights, dim=-1)
    else:
        weights = 1.0 + torch.tanh(weights)
    return torch.matmul(weights, v)


# Although there are already implementation of Multihead Attention in PyTorch, 
# it can be applied when dimension of query is not same with embed dimension.
class MultiheadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, scale=1.0, normalize=False, num_heads=8):
        super(MultiheadAttention, self).__init__()

        self.scale = scale
        self.normalize = normalize
        self.num_heads = num_heads

        embed_dim = v_dim // num_heads
        self.layers = nn.ModuleDict({"q_proj": nn.ModuleList([nn.Linear(q_dim, embed_dim) for _ in range(num_heads)]),
                                     "k_proj": nn.ModuleList([nn.Linear(k_dim, embed_dim) for _ in range(num_heads)]),
                                     "v_proj": nn.ModuleList([nn.Linear(v_dim, embed_dim) for _ in range(num_heads)]),
                                     "out_layer": nn.Linear(v_dim, v_dim, bias=False)})

    def forward(self, q, k, v):
        multi_head_embed = torch.cat([dot_product_attention(self.layers["q_proj"][i](q), self.layers["k_proj"][i](k), self.layers["v_proj"][i](v), 
                                                            scale=self.scale, normalize=self.normalize) for i in range(self.num_heads)], dim=-1)
        return self.layers["out_layer"](multi_head_embed)


class Attention(nn.Module):
    def __init__(self, x_dim, h_dim, attn_type, scale=1.0, normalize=True, num_heads=8):
        super(Attention, self).__init__()
        """
            Attention Module
            embed_dim: dimension of embedding(value)
            attention_type: type of attention. One of the following:
                ['uniform','laplace','dot_product','multihead']
            scale: scale of attention.
            normalize: Boolean determining whether to:
                1. apply softmax to weights so that they sum to 1 across context pts or
                2. apply custom transformation to have weights in [0,1].
            num_heads: number of heads for multihead.
        """
        self.attn_type = attn_type
        self.scale = scale
        self.normalize = normalize
        if self.attn_type == 'multihead':
            self.multihead_attention = MultiheadAttention(x_dim, x_dim, h_dim, scale=scale, normalize=normalize, num_heads=num_heads)

    def forward(self, q, k, v):
        """
            Apply attention to create aggregated representation of r.
            q: (batch_size * num_queries * q_dim)
            k: (batch_size * num_keys * k_dim)
            v: (batch_size * num_values * v_dim)
            num_heads: number of heads. Should divide v_dim.

            output: (batch-size * num_queries * v_dim)
        """
        match self.attn_type:
            case "uniform":
                return uniform_attention(q, v)
            case "laplace":
                return laplace_attention(q, k, v, scale=self.scale, normalize=self.normalize)
            case "dot_product":
                return dot_product_attention(q, k, v, scale=self.scale, normalize=self.normalize)
            case "multihead":
                return self.multihead_attention(q, k, v)