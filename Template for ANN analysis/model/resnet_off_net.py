from torch import nn
import pytorch_lightning as pl
from .common import resnet_cifar
from pipeline.utils import Network
from torchvision.models import resnet18
import torch.nn.functional as F



class ResnetOffNet(pl.LightningModule):
    def __init__(self, depth=20, width_multiplier=1, output_size=10):
        super().__init__()
        self.__dict__.update(locals())
        # set up layers
        self.network = resnet18(pretrained=False, num_classes=output_size, progress=True)
        self.network.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.network.maxpool = nn.Identity()

    def forward(self, x):
        output = self.network(x)
        # output = F.log_softmax(output, dim=-1)
        return output
