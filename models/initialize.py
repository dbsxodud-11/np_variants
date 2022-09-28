
import torch.nn as nn

def init_weight(weight, initialization):
    if initialization == "xavier_uniform":
        nn.init.xavier_uniform_(weight)
    elif initialization == "xavier_normal":
        nn.init.xavier_normal_(weight)
    else:
        pass

def init_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        NotImplementedError()