
import torch
from torch.distributions import MultivariateNormal


class GPCurve:
    def __init__(self, kernel, lb, ub, dtype, device):
        if kernel == "rbf":
            self.kernel = RBFKernel()

        self.lb = lb
        self.ub = ub

        self.dtype = dtype
        self.device = device
        
    def sample(self, batch_size, num_samples, dim):
        """
            Samples Data from Gaussian Process
            
            Output: (batch_size * num_samples * dim), (batch_size * num_samples * 1)
        """

        batch_x = self.lb + (self.ub - self.lb) * torch.rand(batch_size, num_samples, dim).to(dtype=self.dtype, device=self.device)

        mean = torch.zeros(batch_size, num_samples).to(dtype=self.dtype, device=self.device)
        cov = self.kernel(batch_x)
        batch_y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)

        return batch_x, batch_y

    def sample_for_visualize(self, batch_size, num_samples):
        """
            Sample Data from Gaussian Process for visualization
            Only available in 1-dimension setting
        """
        x = torch.linspace(self.lb, self.ub, num_samples).view(1, num_samples, 1).repeat(batch_size, 1, 1).to(dtype=self.dtype, device=self.device)
        mean = torch.zeros(batch_size, num_samples).to(dtype=self.dtype, device=self.device)
        cov = self.kernel(x)
        y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)

        return x, y


class RBFKernel(object):
    def __init__(self, sigma_eps=1e-3, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    def __call__(self, x):
        """
            Return Kernel Matrix
            
            x: (batch_size * num_samples * dim)

            Output: (batch_size * num_samples * num_samples)
        """
        length = 0.1 + (self.max_length - 0.1) * torch.rand(x.shape[0], 1, 1, 1).to(dtype=x.dtype, device=x.device)
        scale = 0.1 + (self.max_scale - 0.1) * torch.rand(x.shape[0], 1, 1).to(dtype=x.dtype, device=x.device)

        dist = (x.unsqueeze(-2) - x.unsqueeze(-3)) / length

        cov = scale.pow(2) * torch.exp(-0.5 * dist.pow(2).sum(-1)) + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(dtype=x.dtype, device=x.device)
        return cov


if __name__ == "__main__":
    dim = 20
    batch_size = 16
    num_samples = 40
    
    dtype = torch.double
    device = torch.device("cuda")

    generator = GPCurve(kernel="rbf", lb=-2.0, ub=2.0, dtype=dtype, device=device)
    batch_x, batch_y = generator.sample(batch_size, num_samples, dim)
    print(batch_x.shape, batch_y.shape)