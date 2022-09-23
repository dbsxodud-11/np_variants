
import torch
import torch.nn as nn
from torch.distributions import Normal


class DeterministicEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, h_dim):
        super(DeterministicEncoder, self).__init__()

        self.model = nn.Sequential(nn.Linear(x_dim + y_dim, h_dim),
                                   nn.ReLU(),
                                   nn.Linear(h_dim, h_dim),
                                   nn.ReLU(),
                                   nn.Linear(h_dim, r_dim))

    def forward(self, x, y):
        """
            x: (batch_size * num_samples * x_dim)
            y: (batch_size * num_samples * y_dim)

            output: (batch_size * r_dim)
        """
        x = torch.cat([x, y], dim=-1)
        return self.model(x).mean(dim=1)


class LatentEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, h_dim):
        super(LatentEncoder, self).__init__()

        self.model = nn.Sequential(nn.Linear(x_dim + y_dim, h_dim),
                                   nn.ReLU(),
                                   nn.Linear(h_dim, h_dim),
                                   nn.ReLU())
        self.mu_head = nn.Linear(h_dim, z_dim)
        self.sigma_head = nn.Linear(h_dim, z_dim)

    def forward(self, x, y):
        """
            x: (batch_size * num_samples * x_dim)
            y: (batch_size * num_samples * y_dim)

            output: Normal(batch_size * z_dim)
        """
        x = torch.cat([x, y], dim=-1)
        x = self.model(x)

        mu = self.mu_head(x).mean(dim=1)
        log_var = self.sigma_head(x).mean(dim=1)
        sigma = torch.exp(0.5 * log_var)
        return Normal(mu, sigma)


class Decoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(nn.Linear(x_dim + r_dim + z_dim, h_dim),
                                   nn.ReLU(),
                                   nn.Linear(h_dim, h_dim),
                                   nn.ReLU())
        self.mu_head = nn.Linear(h_dim, y_dim)
        self.sigma_head = nn.Linear(h_dim, y_dim)

    def forward(self, x, r, z):
        """
            x: (batch_size * num_samples * x_dim)
            r: (batch_size * r_dim)
            z: (batch_size * z_dim)

            output: (batch_size * nun_samples * y_dim)
        """
        num_samples = x.shape[1]
        r = r.unsqueeze(1).repeat(1, num_samples, 1)
        z = z.unsqueeze(1).repeat(1, num_samples, 1)

        x = torch.cat([x, r, z], dim=-1)
        x = self.model(x)

        mu = self.mu_head(x)
        log_var = self.sigma_head(x)
        sigma = torch.exp(0.5 * log_var)
        return Normal(mu, sigma)    


        