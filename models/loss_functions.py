
import torch
from torch.distributions.kl import kl_divergence

def np_loss_func(y_pred, y_target, q_context, q_target):
    likelihood = y_pred.log_prob(y_target).sum((-2, -1)).mean(-1)
    kl = kl_divergence(q_context, q_target).mean(dim=1).sum()
    return -likelihood + kl

def bnp_loss_func(y_pred, y_target):
    likelihood = y_pred.log_prob(y_target).mean(dim=0).sum()
    return -likelihood