
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

def init_weight(weight, initialization):
    match initialization:
        case "xavier_uniform":
            nn.init.xavier_uniform_(weight)
        case "xavier_normal":
            nn.init.xavier_normal_(weight)
        case _:
            pass

def init_activation(activation):
    match activation:
        case "relu":
            return nn.ReLU()
        case "tanh":
            return nn.Tanh()
        case "sigmoid":
            return nn.Sigmoid()
        case "elu":
            return nn.ELU()
        case _:
            NotImplementedError()


class BaseMLP(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 hidden_dims=[64] * 3, activation="relu", weight_initialization=None, bias=True):
        super(BaseMLP, self).__init__()

        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip([input_dim] + hidden_dims, hidden_dims + [output_dim]):
            layer = nn.Linear(in_dim, out_dim, bias=bias)
            init_weight(layer.weight, weight_initialization)
            self.layers.append(layer)

        self.activation = init_activation(activation)  

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x


class DeterministicEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, h_dim, attention=False):
        super(DeterministicEncoder, self).__init__()

        self.model = BaseMLP(x_dim + y_dim, r_dim, hidden_dims=[h_dim]*3, activation="relu",
                                                   weight_initialization="xavier_uniform")

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

        self.model = BaseMLP(x_dim + y_dim, h_dim, hidden_dims=[h_dim]*2, activation="relu",
                                                   weight_initialization="xavier_uniform")
        self.mu_head = nn.Linear(h_dim, z_dim)
        init_weight(self.mu_head.weight, initialization="xavier_uniform")
        
        self.sigma_head = nn.Linear(h_dim, z_dim)
        init_weight(self.sigma_head.weight, initialization="xavier_uniform")

    def forward(self, x, y):
        """
            x: (batch_size * num_samples * x_dim)
            y: (batch_size * num_samples * y_dim)

            output: Normal(batch_size * z_dim)
        """
        x = torch.cat([x, y], dim=-1)
        x = self.model(x)

        mu = self.mu_head(x).mean(dim=1)
        sigma = 0.1 + 0.9 * torch.sigmoid(self.sigma_head(x).mean(dim=1))
        return Normal(mu, sigma)


class Decoder(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim):
        super(Decoder, self).__init__()

        self.model = BaseMLP(x_dim + r_dim + z_dim, h_dim, hidden_dims=[h_dim]*2, activation="relu",
                                                           weight_initialization="xavier_uniform")
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
        sigma = 0.1 + 0.9 * F.softplus(self.sigma_head(x))
        return Normal(mu, sigma)    


        