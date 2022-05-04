import torch.nn as nn
import torch.nn.functional as F
from pipeline.utils import Add, Identity


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


def res_block(in_channels, out_channels, stride):
    need_proj = (in_channels != out_channels) or (stride != 1)
    return {
        'prep': nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU()),
        'branch': (nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        ), 'prep'),
        'proj': (nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
                 if need_proj else Identity(), 'prep'),
        'add': (Add(), ['proj', 'branch'])
    }


def get_model_def(channels=(64, 128, 256, 256)):
    return {
        'input': None,
        'preprocess': nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        ),
        'group1': {
            'block1': res_block(channels[0], channels[0], stride=1),
            'block2': res_block(channels[0], channels[0], stride=1),
        },
        'group2': {
            'block1': res_block(channels[0], channels[1], stride=2),
            'block2': res_block(channels[1], channels[1], stride=1),
        },
        'group3': {
            'block1': res_block(channels[1], channels[2], stride=2),
            'block2': res_block(channels[2], channels[2], stride=1),
        },
        'group4': {
            'block1': res_block(channels[2], channels[3], stride=2),
            'block2': res_block(channels[3], channels[3], stride=1),
        },
        'pool': nn.MaxPool2d(4),
        'proj': nn.Sequential(nn.Flatten(),
                              nn.Linear(channels[3], 10))
    }


def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, relu=True):
    return {
        'conv': nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
        'bn': nn.BatchNorm2d(out_channels),
        'relu': nn.ReLU()
    } if relu else {
        'conv': nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
        'bn': nn.BatchNorm2d(out_channels)
    }


def resnet_cifar(depth=20,
                 width_multiplier=1,
                 num_classes=10,
                 input_shape=(3, 32, 32)):
    # Inspired by https://github.com/google-research/google-research/blob/master/do_wide_and_deep_networks_learn_the_same_things/resnet_cifar.py#L100
    assert (depth - 2) % 6 == 0, "Depth-2 must be a multiple of 6, as in resnet-18, resnet-24, resnet-30, etc"
    num_features = 16 * int(width_multiplier)
    num_blocks = (depth - 2) // 6

    model_spec = {
        'input': None,
        'layer000': conv_bn(input_shape[0], num_features)
    }

    layer_count, prev_layer_features = 1, num_features
    for stack in range(3):
        for block in range(num_blocks):
            if block == 0 and stack > 0:
                # If this is first layer of the block, we downsample space (stride=2) and upsample features.
                # Because of this change in shape, the residual block requires an extra 'projection' step.
                num_features *= 2
                res_block = {
                    'x': Identity(),
                    'block0': conv_bn(num_features // 2, num_features, stride=2),
                    'block1': conv_bn(num_features, num_features, stride=1, relu=False),
                    'proj_x': (nn.Conv2d(num_features // 2, num_features, kernel_size=1, stride=2), 'x'),
                    'skip': (Add(), ['proj_x', -1]),
                    'relu': nn.ReLU()
                }
            else:
                # Just contintuing to add layers to this block, all with same #features
                res_block = {
                    'x': Identity(),
                    'block0': conv_bn(num_features, num_features, stride=1),
                    'block1': conv_bn(num_features, num_features, stride=1, relu=False),
                    'skip': (Add(), ['x', 'block1/bn']),
                    'relu': nn.ReLU()
                }
            # Add this 'block' to the network with a name like 'layer018'
            model_spec[f'layer{layer_count:03d}'] = res_block
            layer_count += 1

    # Add spatial pooling and projection to classes
    model_spec['pool'] = nn.Sequential(nn.AvgPool2d(8), nn.Flatten())
    model_spec['logits'] = nn.Linear(num_features, num_classes)
    model_spec['prob'] = nn.Softmax(dim=-1)

    return model_spec


# **********************************************************************************************************************