import torch
from torch import nn

class StandardNet(nn.Module):
    """ If you want to use pretrained model, or simply the standard structure implemented
        by Pytorch official, please use this template. It enable you to easily control whether
        use or not the pretrained weights, and whether to freeze the internal layers or not,
        and the in/out channel numbers, resnet version. This is made for resnet, but you can
        also adapt it to other structures by changing the `torch.hub.load` content.
    """
    def __init__(self, in_channel=3, out_channel=10, resnet_name='resnet18', freeze=False, pretrained=False):
        super().__init__()
        print(in_channel, out_channel)
        self.resnet = torch.hub.load('pytorch/vision:v0.9.0', resnet_name, pretrained=pretrained)
        
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        inter_ftrs = self.resnet.conv1.out_channels
        self.resnet.conv1 = nn.Conv2d(in_channel, inter_ftrs, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, out_channel)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x
