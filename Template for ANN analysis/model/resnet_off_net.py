from torch import nn
import pytorch_lightning as pl
from .common import resnet_cifar
from pipeline.utils import Network
from torchvision.models import resnet18



class ResnetOffNet(pl.LightningModule):
    def __init__(self, depth=20, width_multiplier=1, output_size=10):
        super().__init__()
        self.__dict__.update(locals())
        # set up layers
        self.network = resnet18(pretrained=False, progress=True)

    def forward(self, x):
        output = self.network(x)
        return output
