import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_group_norm: bool = False,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes) if not use_group_norm else nn.GroupNorm(groups, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes) if not use_group_norm else nn.GroupNorm(groups, planes)
        self.downsample = downsample
        self.stride = stride

    expansion: int = 1

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_group_norm: bool = False,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width) if not use_group_norm else nn.GroupNorm(groups, width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width) if not use_group_norm else nn.GroupNorm(groups, width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(width) if not use_group_norm else nn.GroupNorm(groups, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=2, use_group_norm=False):
        super(ResNetMultiImageInput, self).__init__(block, layers)

        self.groups = 1 if not use_group_norm else 32

        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if not use_group_norm else nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], use_group_norm=use_group_norm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_group_norm=use_group_norm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_group_norm=use_group_norm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_group_norm=use_group_norm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, use_group_norm=False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion) if not use_group_norm else nn.GroupNorm(self.groups, planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, use_group_norm=use_group_norm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, use_group_norm=use_group_norm))

        return nn.Sequential(*layers)

def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, use_group_norm=False):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 4, 36, 3]}[num_layers]
    block_type = {18: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, use_group_norm=use_group_norm)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        # key：强行将预训练权重复制一份
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)

    del model.fc
    del model.avgpool
    return model


class ResnetUnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=2, use_group_norm=False):
        super(ResnetUnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, use_group_norm)

        # encoder channel * 4
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, x):
        self.features = []

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        self.features.append(self.encoder.relu(x)) # /2
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1]))) # /4
        self.features.append(self.encoder.layer2(self.features[-1])) # /8
        self.features.append(self.encoder.layer3(self.features[-1])) # /16
        self.features.append(self.encoder.layer4(self.features[-1])) # /32

        return self.features

# if __name__=='__main__':
#     dummy_input = torch.zeros(1,6,256,256)
#     encoder = ResnetUnetEncoder(101,False)
#     print(encoder)
#     out = encoder(dummy_input)
#     print(encoder.encoder.layer4[-1])
#     print(encoder.encoder.layer4[-1].children())
#     for i in encoder.encoder.layer4[-1].children():
#         print(i)

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained=False, num_input_images=2, use_group_norm=False):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, use_group_norm)

        # encoder channel * 4
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, x):
        self.features = []

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        x = self.encoder.relu(x) # /2
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x) # /4
        x = self.encoder.layer2(x) # /8
        x = self.encoder.layer3(x) # /16
        x = self.encoder.layer4(x) # /32

        return x