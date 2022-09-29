import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, out_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim

        self.model = nn.ModuleDict({"q_proj": nn.Linear(q_dim, out_dim, bias=False),
                                    "k_proj": nn.Linear(k_dim, out_dim, bias=False),
                                    "v_proj": nn.Linear(v_dim, out_dim, bias=False),
                                    "out_layer": nn.Linear(out_dim, out_dim),
                                    "ln1": nn.LayerNorm(out_dim),
                                    "ln2": nn.LayerNorm(out_dim)})

    def scatter(self, x):
        return torch.cat(x.chunk(self.num_heads, dim=-1), dim=-3)

    def gather(self, x):
        return torch.cat(x.chunk(self.num_heads, dim=-3), dim=-1)

    def attend(self, q, k, v, mask=None):
        q_, k_, v_ = [self.scatter(x) for x in [q, k, v]]
        A_logits = q_ @ k_.transpose(-2, -1) / math.sqrt(self.out_dim)
        if mask is not None:
            mask = mask.bool().to(dtype=q.dtype, device=q.device)
            mask = torch.stack([mask] * q.shape[-2], dim=-2)
            mask = torch.cat([mask] * self.num_heads, dim=-3)
            A = torch.softmax(A_logits.masked_fill(mask, -float("inf")), dim=-1)
            A = A.masked_fill(torch.isnan(A), 0.0)
        else:
            A = torch.softmax(A_logits, -1)
        return self.gather(A @ v_)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.model["q_proj"](q), self.model["k_proj"](k), self.model["v_proj"](v)
        out = self.model["ln1"](q + self.attend(q, k, v, mask=mask))
        out = self.model["ln2"](out + F.relu(self.model["out_layer"](out)))
        return out

class SelfAttention(MultiheadAttention):
    def __init__(self, input_dim, output_dim, num_heads=8):
        super().__init__(input_dim, input_dim, input_dim, output_dim, num_heads)

    def forward(self, x, mask=None):
        return super().forward(x, x, x, mask=mask)