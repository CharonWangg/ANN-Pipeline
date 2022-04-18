import torch.nn as nn

# activation function parser
def configure_activation(activation):
    activation = activation.lower()
    if activation == "relu":
        activation = nn.ReLU()
    elif activation == "leaky_relu":
        activation = nn.LeakyReLU()
    elif activation == "tanh":
        activation = nn.Tanh()
    elif activation == "sigmoid":
        activation = nn.Sigmoid()
    elif activation == "none":
        activation = nn.Identity()
    return activation


