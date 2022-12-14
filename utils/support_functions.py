
import numpy as np
import torch

def context_target_split(x, y, num_context, num_target):
    idx = np.random.choice(x.shape[1], size=num_context + num_target, replace=False)
    
    x_context = x[:, idx[:num_context], :]
    y_context = y[:, idx[:num_context], :]

    x_target = x[:, idx, :]
    y_target = y[:, idx, :]
    
    return x_context, y_context, x_target, y_target

def sample_with_replacement(*items, num_samples):
    items_resample = [[] for _ in range(len(items))]

    for i in range(len(items)):
        for _ in range(num_samples):
            idx = np.random.choice(items[i].shape[-2], size=items[i].shape[-2], replace=True)
            items_resample[i].append(items[i][..., idx, :])

    return [torch.stack(item_resample, dim=0) for item_resample in items_resample]

def stack_tensor(x, num_samples=1, dim=0):
    return torch.stack([x] * num_samples, dim=dim)