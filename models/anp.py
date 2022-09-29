import math

import torch
import torch.nn as nn
from attrdict import AttrDict

from utils.support_functions import stack_tensor
from models.np import NP
from models.building_blocks import CrossAttentionEncoder, LatentEncoder, Decoder

class ANP(NP):
    def __init__(self, x_dim, y_dim, r_dim=128, z_dim=128, h_dim=128,
                       enc_pre_num_layers=4, enc_post_num_layers=2, 
                       qk_num_layers=4, v_num_layers=2, dec_num_layers=3, 
                       self_attn=True):
        super().__init__(x_dim, y_dim, r_dim, z_dim, h_dim,
                         enc_pre_num_layers, enc_post_num_layers, dec_num_layers)

        self.deterministic_encoder = CrossAttentionEncoder(x_dim, y_dim, r_dim, h_dim,
                                                           qk_num_layers=qk_num_layers,
                                                           v_num_layers=v_num_layers,
                                                           self_attn=self_attn)

    def forward(self, batch, num_samples=1):
        out = AttrDict()
        num_context, num_target = batch.x_context.shape[-2], batch.x_target.shape[-2]

        r = self.deterministic_encoder(batch.x_context, batch.y_context, batch.x_target)
        r = stack_tensor(r, num_samples, dim=0)

        qz_context = self.latent_encoder(batch.x_context, batch.y_context)

        if self.training:
            qz_target = self.latent_encoder(batch.x_target, batch.y_target)
            z = qz_target.rsample([num_samples])
        else:
            z = qz_context.rsample([num_samples])

        x_target = stack_tensor(batch.x_target, num_samples, dim=0)
        enc = torch.cat([r, stack_tensor(z, num_target, dim=-2)], dim=-1)

        p_y_pred = self.decoder(x_target, enc)
        y_target = stack_tensor(batch.y_target, num_samples, dim=0)
        
        if self.training:
            recon = p_y_pred.log_prob(y_target).sum(dim=-1).mean(dim=-1)
            log_qz_target = qz_target.log_prob(z).sum(dim=-1)
            log_qz_context = qz_context.log_prob(z).sum(dim=-1)

            log_weight = recon + log_qz_context - log_qz_target
            weighted_loss = torch.logsumexp(log_weight, dim=0) - math.log(log_weight.shape[0])
            out.loss = -weighted_loss.mean()
        else:
            likelihood = p_y_pred.log_prob(y_target).sum(dim=-1)
            out.context_likelihood = likelihood[..., :num_context].mean()
            out.target_likelihood = likelihood[..., num_context:].mean()
        
        return out