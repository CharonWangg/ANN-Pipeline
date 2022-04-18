import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from configure_activation import configure_activation
from . import common


class FcnNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=[32, 32],
                 output_size=10, num_hidden_layers=2, dropout=0.5,
                 activation=nn.ReLU):
        super().__init__()
        self.__dict__.update(locals())
        # set up activation
        self.activation = configure_activation(activation)
        # input layer with activation
        self.input_layer = nn.Sequential(nn.Linear(self.input_size, self.hidden_size[0]),
                                         self.activation,)
        # hidden layer
        self.hidden_layers = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_size[i], self.hidden_size[i+1]),
                                                          self.activation) for i in range(len(self.hidden_size) - 1)])
        # output layer
        self.output_layer = nn.Linear(self.hidden_size[-1], self.output_size)

def forward(self, x):
        x = x.view(x.size(0), -1)

        # input layer
        x = self.input_layer(x)
        hs = {"h_0": x.cpu()}
        # hidden layers
        for i in range(len(self.hidden_layers)):
            x = F.dropout(self.hidden_layers[i](x), p=self.dropout, training=self.training)
            hs[f"h_{i + 1}"] = x.cpu()
        # output layer
        x = self.output_layer(x)
        hs["h_o"] = x.cpu()
        # softmax layer
        hs["h_p"] = F.softmax(x, dim=-1).cpu()

        return hs, x
