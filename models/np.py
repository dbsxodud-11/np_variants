
import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence

from models.building_blocks import DeterministicEncoder, LatentEncoder, Decoder


class NeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim=128, r_dim=128, z_dim=128, num_train_samples=4, num_test_samples=50):
        super(NeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim

        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

        self.deterministic_encoder = DeterministicEncoder(x_dim, y_dim, r_dim, h_dim, attn_type="uniform", self_attn=False)
        self.latent_encoder = LatentEncoder(x_dim, y_dim, z_dim, h_dim, attn_type="uniform", self_attn=False)
        self.decoder = Decoder(x_dim, y_dim, r_dim, z_dim, h_dim)

    def forward(self, x_context, y_context, x_target, y_target=None):
        if y_target is not None:
            x_context = x_context.unsqueeze(0).repeat(self.num_train_samples, 1, 1, 1)
            y_context = y_context.unsqueeze(0).repeat(self.num_train_samples, 1, 1, 1)

            x_target = x_target.unsqueeze(0).repeat(self.num_train_samples, 1, 1, 1)
            y_target = y_target.unsqueeze(0).repeat(self.num_train_samples, 1, 1, 1)

            q_context = self.latent_encoder(x_context, y_context)
            q_target = self.latent_encoder(x_target, y_target)

            r_context = self.deterministic_encoder(x_context, y_context, x_target).unsqueeze(-2).repeat(1, 1, x_target.shape[-2], 1)
            z_context = q_context.rsample().unsqueeze(-2).repeat(1, 1, x_target.shape[-2], 1)

            y_pred = self.decoder(x_target, r_context, z_context)

            log_prob = y_pred.log_prob(y_target).sum((-2, -1)).mean(-1)
            kl_div = kl_divergence(q_target, q_context).sum(-1).mean(-1)

            loss = torch.logsumexp(-log_prob + kl_div, dim=0)
            return loss
        else:
            x_context = x_context.unsqueeze(0).repeat(self.num_test_samples, 1, 1, 1)
            y_context = y_context.unsqueeze(0).repeat(self.num_test_samples, 1, 1, 1)

            x_target = x_target.unsqueeze(0).repeat(self.num_test_samples, 1, 1, 1)
            
            q_context = self.latent_encoder(x_context, y_context)

            r_context = self.deterministic_encoder(x_context, y_context, x_target).unsqueeze(-2).repeat(1, 1, x_target.shape[-2], 1)
            z_context = q_context.rsample().unsqueeze(-2).repeat(1, 1, x_target.shape[-2], 1)

            y_pred = self.decoder(x_target, r_context, z_context)
            return y_pred


if __name__ == "__main__":
    model = NeuralProcess(x_dim=1, y_dim=1)