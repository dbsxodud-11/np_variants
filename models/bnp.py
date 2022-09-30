import math

import torch
from attrdict import AttrDict

from utils.support_functions import sample_with_replacement, stack_tensor
from models.cnp import CNP


class BNP(CNP):
    def __init__(self, x_dim, y_dim, r_dim=128, h_dim=128,
                       enc_pre_num_layers=4, enc_post_num_layers=2, dec_num_layers=3,
                       bootstrap=True):
        super().__init__(x_dim, y_dim, r_dim, h_dim,
                         enc_pre_num_layers, enc_post_num_layers, dec_num_layers, bootstrap)

    def encode(self, x_context, y_context, num_target):
        h1 = self.deterministic_encoder1(x_context, y_context)
        h2 = self.deterministic_encoder2(x_context, y_context)
        return stack_tensor(torch.cat([h1, h2], dim=-1), num_target, dim=-2)

    def compute_loss(self, p_y_pred, y_target):
        log_weight = p_y_pred.log_prob(y_target).sum(dim=-1).mean(dim=-1)
        weighted_loss = torch.logsumexp(log_weight, dim=0) - math.log(log_weight.shape[0])
        return -weighted_loss.mean()

    def forward(self, batch, num_samples=1):
        out = AttrDict()
        num_context, num_target = batch.x_context.shape[-2], batch.x_target.shape[-2]
        
        with torch.no_grad():
            x_bootstrap, y_bootstrap = sample_with_replacement(batch.x_context, batch.y_context,
                                                               num_samples=num_samples)
            x_context = stack_tensor(batch.x_context, num_samples, dim=0)
            y_context = stack_tensor(batch.y_context, num_samples, dim=0)                                                   
            h = self.encode(x_bootstrap, y_bootstrap, num_context)
            p_y_pred = self.decoder(x_context, h)

            residual = (y_context - p_y_pred.loc) / p_y_pred.scale
            residual_bootstrap = sample_with_replacement(residual, num_samples=1)[0].squeeze(dim=0)

            x_context_bootstrap = x_context
            y_context_bootstrap = p_y_pred.loc + p_y_pred.scale * residual_bootstrap

        h_base = self.encode(batch.x_context, batch.y_context, num_target)

        x_target = stack_tensor(batch.x_target, num_samples, dim=0)
        h_bootstrap = self.encode(x_context_bootstrap, y_context_bootstrap, num_target)

        p_y_pred_base = self.decoder(batch.x_target, h_base)
        p_y_pred_bootstrap = self.decoder(x_target, stack_tensor(h_base, num_samples, dim=0), extra_enc=h_bootstrap)
        y_target = stack_tensor(batch.y_target, num_samples, dim=0)

        if self.training:
            loss_base = self.compute_loss(p_y_pred_base, batch.y_target)
            loss_bootstrap = self.compute_loss(p_y_pred_bootstrap, y_target)
            out.loss = loss_base + loss_bootstrap
        else:
            likelihood = p_y_pred_bootstrap.log_prob(y_target).sum(dim=-1)
            out.context_likelihood = likelihood[..., :num_context].mean()
            out.target_likelihood = likelihood[..., num_context:].mean()
        
        return out