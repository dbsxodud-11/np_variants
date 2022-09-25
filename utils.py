
import numpy as np
import torch

def context_target_split(x, y, num_context, num_extra_target):
    idx = np.random.choice(x.shape[1], size=num_context + num_extra_target, replace=False)
    
    x_context = x[:, idx[:num_context], :]
    y_context = y[:, idx[:num_context], :]

    x_target = x[:, idx, :]
    y_target = y[:, idx, :]
    
    return x_context, y_context, x_target, y_target

def sample_with_replacement(*items, num_samples):
    x_resample, y_resample = [], []
    items_resample = [[] for _ in range(len(items))]

    for i in range(len(items)):
        for _ in range(num_samples):
            idx = np.random.choice(items[i].shape[1], size=items[i].shape[1], replace=True)
            items_resample[i].append(items[i][:, idx, :])

    return [torch.cat(item_resample, dim=0) for item_resample in items_resample]