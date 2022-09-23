import numpy as np

def context_target_split(x, y, num_context, num_extra_target):
    idx = np.random.choice(x.shape[1], size=num_context + num_extra_target, replace=False)
    
    x_context = x[:, idx[:num_context], :]
    y_context = y[:, idx[:num_context], :]

    x_target = x[:, idx, :]
    y_target = y[:, idx, :]

    return x_context, y_context, x_target, y_target