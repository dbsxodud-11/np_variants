
import torch
import torch.nn as nn

from utils import sample_with_replacement
from models.building_blocks import DeterministicEncoder, LatentEncoder, Decoder


class BootstrappingNeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim=128, r_dim=128, train_num_samples=4, test_num_samples=50):
        super(BootstrappingNeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        self.train_num_samples = train_num_samples
        self.test_num_samples = test_num_samples

        self.deterministic_encoder_base = DeterministicEncoder(x_dim, y_dim, r_dim, h_dim, attn_type="uniform", self_attn=False)
        self.deterministic_encoder_bootstrap = DeterministicEncoder(x_dim, y_dim, r_dim, h_dim, attn_type="uniform", self_attn=False)
        self.decoder = Decoder(x_dim, y_dim, r_dim, r_dim, h_dim)

    def forward(self, x_context, y_context, x_target, y_target=None):
        batch_size = x_context.shape[0]

        r_context_base = self.deterministic_encoder_base(x_context, y_context, x_target)
        
        if y_target is not None:
            x_context_bootstrap, y_context_bootstrap = self.get_bootstrap_context(x_context, y_context, self.train_num_samples)
        else:
            x_context_bootstrap, y_context_bootstrap = self.get_bootstrap_context(x_context, y_context, self.test_num_samples)

        r_context_bootstrap = self.deterministic_encoder_bootstrap(x_context_bootstrap, y_context_bootstrap, x_target.repeat(self.train_num_samples, 1, 1))
        _, num_context, _ = r_context_bootstrap.size()
        r_context_bootstrap = r_context_bootstrap.view(-1, batch_size, num_context, self.r_dim).mean(dim=0)

        y_pred = self.decoder(x_target, r_context_base, r_context_bootstrap)
        return y_pred

    def get_bootstrap_context(self, x_context, y_context, num_samples):
        x_paired_bootstrap, y_paired_bootstrap = sample_with_replacement(x_context, y_context, num_samples=num_samples)
        x_context, y_context = x_context.repeat(num_samples, 1, 1), y_context.repeat(num_samples, 1, 1)

        r_context_base = self.deterministic_encoder_base(x_paired_bootstrap, y_paired_bootstrap, x_context)
        r_context_bootstrap = self.deterministic_encoder_bootstrap(x_paired_bootstrap, y_paired_bootstrap, x_context)

        y_pred = self.decoder(x_context, r_context_base, r_context_bootstrap)
        residual = (y_context - y_pred.loc) / y_pred.scale
        residual_bootstrap = sample_with_replacement(residual, num_samples=1)[0]

        x_context_bootstrap = x_context
        y_context_bootstrap = y_pred.loc + y_pred.scale * residual_bootstrap
        return x_context_bootstrap, y_context_bootstrap