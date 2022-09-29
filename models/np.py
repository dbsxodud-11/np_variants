
import torch
import torch.nn as nn

from models.building_blocks import DeterministicEncoder, LatentEncoder, Decoder
from utils.support_functions import stack_tensor


class NeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim=128, z_dim=128, h_dim=128, enc_pre_num_layers=4, enc_post_num_layers=2, dec_num_layers=3):
        super(NeuralProcess, self).__init__()

        self.deterministic_encoder = DeterministicEncoder(x_dim, y_dim, r_dim, h_dim, 
                                                          pre_num_layers=enc_pre_num_layers,
                                                          post_num_layers=enc_post_num_layers, 
                                                          self_attn=False)

        self.latent_encoder = LatentEncoder(x_dim, y_dim, z_dim, h_dim, 
                                            pre_num_layers=enc_pre_num_layers,
                                            post_num_layers=enc_post_num_layers, 
                                            self_attn=False)

        self.decoder = Decoder(x_dim, y_dim, r_dim + z_dim, h_dim, num_layers=dec_num_layers)

    def forward(self, x_context, y_context, x_target, y_target=None, num_samples=1):
        if y_target is not None:
            qz_context = self.latent_encoder(x_context, y_context)
            qz_target = self.latent_encoder(x_target, y_target)

            r_context = stack_tensor(self.deterministic_encoder(x_context, y_context), num_samples, dim=0)
            z = qz_target.rsample([num_samples])

            x_target = stack_tensor(x_target, num_samples, dim=0)
            enc = stack_tensor(torch.cat([r_context, z], dim=-1), x_target.shape[-2], dim=-2)
            p_y_pred = self.decoder(x_target, enc)

            return p_y_pred, qz_context, qz_target
        else:
            qz_context = self.latent_encoder(x_context, y_context)

            r_context = stack_tensor(self.deterministic_encoder(x_context, y_context), num_samples, dim=0)
            z = qz_context.rsample([num_samples])

            x_target = stack_tensor(x_target, num_samples, dim=0)
            enc = stack_tensor(torch.cat([r_context, z], dim=-1), x_target.shape[-2], dim=-2)
            p_y_pred = self.decoder(x_target, enc)

            return p_y_pred