
from turtle import pos
import torch
import torch.nn as nn
from attrdict import AttrDict

from models.building_blocks import DeterministicEncoder, Decoder


class CNP(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim=128, h_dim=128, enc_pre_num_layers=4, enc_post_num_layers=2, dec_num_layers=3):
        super(CNP, self).__init__()

        self.deterministic_encoder1 = DeterministicEncoder(x_dim, y_dim, r_dim, h_dim, 
                                                           pre_num_layers=enc_pre_num_layers,
                                                           post_num_layers=enc_post_num_layers, 
                                                           self_attn=False)

        # For pair comparsion, we used two identical encoders for CNP to match the number of parameters
        self.deterministic_encoder2 = DeterministicEncoder(x_dim, y_dim, r_dim, h_dim, 
                                                           pre_num_layers=enc_pre_num_layers,
                                                           post_num_layers=enc_post_num_layers, 
                                                           self_attn=False)

        self.decoder = Decoder(x_dim, y_dim, r_dim * 2, h_dim, num_layers=dec_num_layers)

    def forward(self, x_context, y_context, x_target, y_target=None):
        h1 = self.deterministic_encoder1(x_context, y_context)
        h2 = self.deterministic_encoder2(x_context, y_context)
        h = torch.stack([torch.cat([h1, h2], dim=-1)] * x_target.shape[-2], dim=-2)

        p_y_pred = self.decoder(x_target, h)
        return p_y_pred


