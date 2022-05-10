from torch import nn
import pytorch_lightning as pl
from .common import resnet_cifar
from ..utils import Network




class ResnetHeNet(pl.LightningModule):
    def __init__(self, depth=20, width_multiplier=1, output_size=10):
        super().__init__()
        self.__dict__.update(locals())
        # set up layers
        self.network = Network(resnet_cifar(depth=self.depth,
                                            width_multiplier=self.width_multiplier,
                                            num_classes=self.output_size))

    def forward(self, x):
        inp = {"input": x}
        output = self.network(inp)
        return output
