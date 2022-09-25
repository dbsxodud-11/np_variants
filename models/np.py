
import torch.nn as nn

from models.building_blocks import DeterministicEncoder, LatentEncoder, Decoder


class NeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim=128, r_dim=128, z_dim=128):
        super(NeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim

        self.deterministic_encoder = DeterministicEncoder(x_dim, y_dim, h_dim, h_dim, attn_type="uniform", self_attn=False)
        self.latent_encoder = LatentEncoder(x_dim, y_dim, z_dim, h_dim, attn_type="uniform", self_attn=False)
        self.decoder = Decoder(x_dim, y_dim, r_dim, z_dim, h_dim)

    def forward(self, x_context, y_context, x_target, y_target=None):
        if y_target is not None:
            q_context = self.latent_encoder(x_context, y_context)
            q_target = self.latent_encoder(x_target, y_target)

            r_context = self.deterministic_encoder(x_context, y_context, x_target)
            z_context = q_context.rsample()

            y_pred = self.decoder(x_target, r_context, z_context)
            return y_pred, q_context, q_target
        else:
            q_context = self.latent_encoder(x_context, y_context)

            r_context = self.deterministic_encoder(x_context, y_context, x_target)
            z_context = q_context.rsample()
            y_pred = self.decoder(x_target, r_context, z_context)
            return y_pred