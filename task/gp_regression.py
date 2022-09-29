import math
import random

import torch
from torch.distributions import MultivariateNormal
from attrdict import AttrDict


class GP_REGRESSION(object):
    def __init__(self, dim, kernel, lb, ub, sigma_eps=1e-3, max_length=0.6, max_scale=1.0):
        self.dim = dim
        
        if kernel == "rbf":
            self.kernel = RBFKernel(sigma_eps, max_length, max_scale)
        elif kernel == "matern":
            self.kernel = RBFKernel(sigma_eps, max_length, max_scale)
        elif kernel == "periodic":
            self.kernel = RBFKernel(sigma_eps, max_length, max_scale)
        
        self.lb = lb
        self.ub = ub
        
    def sample(self, batch_size, num_samples, device):
        """
            Samples Data from Gaussian Process
            
            Output: (batch_size * num_samples * dim), (batch_size * num_samples * 1)
        """
        batch = AttrDict()
        num_context = random.randint(3, num_samples-3)
        num_target = random.randint(3, num_samples-num_context)

        batch.x_target = self.lb + (self.ub - self.lb) * torch.rand(batch_size, num_context+num_target, self.dim).to(device)

        mean = torch.zeros(batch_size, num_context+num_target).to(device)
        cov = self.kernel(batch.x_target)
        batch.y_target = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)

        batch.x_context = batch.x_target[:, :num_context, :]
        batch.y_context = batch.y_target[:, :num_context, :]

        return batch

    def sample_for_visualize(self, batch_size, num_samples, device):
        """
            Sample Data from Gaussian Process for visualization
            Only available in 1-dimension setting
        """
        x = torch.linspace(self.lb, self.ub, num_samples).view(1, num_samples, 1).repeat(batch_size, 1, 1).to(device)
        mean = torch.zeros(batch_size, num_samples).to(device)
        cov = self.kernel(x)
        y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)

        return x, y


class RBFKernel(object):
    def __init__(self, sigma_eps=1e-3, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        length = 0.1 + (self.max_length - 0.1) \
                * torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale - 0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        # batch_size * num_points * num_points * dim
        dist = (x.unsqueeze(-2) - x.unsqueeze(-3)) / length

        # batch_size * num_points * num_points
        cov = scale.pow(2) * torch.exp(-0.5 * dist.pow(2).sum(-1)) \
                + self.sigma_eps ** 2 * torch.eye(x.shape[-2]).to(x.device)

        return cov

class Matern52Kernel(object):
    def __init__(self, sigma_eps=1e-3, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        length = 0.1 + (self.max_length - 0.1) \
                * torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale - 0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        # batch_size * num_points * num_points
        dist = torch.norm((x.unsqueeze(-2) - x.unsqueeze(-3)) / length, dim=-1)

        cov = scale.pow(2) * (1 + math.sqrt(5.0) * dist + 5.0 * dist.pow(2)/3.0) \
                * torch.exp(-math.sqrt(5.0) * dist) \
                + self.sigma_eps ** 2 * torch.eye(x.shape[-2]).to(x.device)

        return cov

class PeriodicKernel(object):
    def __init__(self, sigma_eps=1e-3, max_length=0.6, max_scale=1.0):
        #self.p = p
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        p = 0.1 + 0.4 * torch.rand([x.shape[0], 1, 1], device=x.device)
        length = 0.1 + (self.max_length - 0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale - 0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        dist = x.unsqueeze(-2) - x.unsqueeze(-3)
        cov = scale.pow(2) * torch.exp(\
                - 2 * (torch.sin(math.pi * dist.abs().sum(-1) / p) / length).pow(2)) \
                + self.sigma_eps ** 2 * torch.eye(x.shape[-2]).to(x.device)

        return cov