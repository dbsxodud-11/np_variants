
import torch.nn as nn

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