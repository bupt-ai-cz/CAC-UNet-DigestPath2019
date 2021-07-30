import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from ..ops.ibn import IBN
from ..registry import BACKBONE

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, ibn=False):
                #  no ibn
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, ibn=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.with_ibn = ibn
        if self.with_ibn:
            self.ibnc = IBN(width)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out) if not self.with_ibn else self.ibnc(out)
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


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, with_ibn=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=with_ibn)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       ibn=with_ibn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       ibn=with_ibn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       ibn=with_ibn)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.out_channels = [64] + [x * block.expansion for x in [64, 128, 256, 512]]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, ibn=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if ibn:
            ibn = False if planes == 512 else True
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, ibn=ibn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, ibn=ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        outs = []
        outs.append(x)          # 1/2
        x = self.maxpool(x)
        
        x = self.layer1(x)
        outs.append(x)          # 1/4

        x = self.layer2(x)
        outs.append(x)          # 1/8

        x = self.layer3(x)
        outs.append(x)          # 1/16   DL: 1/8

        x = self.layer4(x)
        outs.append(x)          # 1/32   DL: 1/8

        # x = self.avgpool(x)
        # x = x.reshape(x.size(0), -1)
        # x = self.fc(x)

        return tuple(outs)


def _resnet(arch, block, layers, pretrained, progress, with_ibn, replace_stride_with_dilation=[False, False, False], **kwargs):
    model = ResNet(block, layers, 
                    with_ibn=with_ibn, 
                    replace_stride_with_dilation=replace_stride_with_dilation,
                    **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

# -----------------------------------------------------------------------------
#  ResNet 
# -----------------------------------------------------------------------------
@BACKBONE.register("R-18-C1-C5")
def build_resnet18(pretrained=False, progress=True,  with_ibn=False, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, 
                    with_ibn=False,
                   **kwargs)

@BACKBONE.register("R-34-C1-C5")
def build_resnet34(pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                    with_ibn=False,
                   **kwargs)

@BACKBONE.register("R-50-C1-C5")
def build_resnet50(pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, with_ibn,**kwargs)

@BACKBONE.register("R-101-C1-C5")
def build_resnet101(pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, with_ibn,**kwargs)

@BACKBONE.register("R-152-C1-C5")
def build_resnet152(pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, with_ibn, **kwargs)

@BACKBONE.register("RX-50-C1-C5")
def build_resnext50_32x4d(pretrained=False, progress=True, with_ibn=False,**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, with_ibn, **kwargs)

@BACKBONE.register("RX-101-C1-C5")
def build_resnext101_32x8d(pretrained=False, progress=True, with_ibn=False,**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, with_ibn, **kwargs)



# -----------------------------------------------------------------------------
# ResNet replace_stride_with_dilation 
# -----------------------------------------------------------------------------
@BACKBONE.register("R-DL-18-C1-C5")
def build_resnet18_aspp(pretrained=False, progress=True,  with_ibn=False, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, with_ibn=False,
                    replace_stride_with_dilation=[False, True, True],
                   **kwargs)

@BACKBONE.register("R-DL-34-C1-C5")
def build_resnet34_aspp(pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, with_ibn=False,
                    replace_stride_with_dilation=[False, True, True],
                   **kwargs)

@BACKBONE.register("R-DL-50-C1-C5")
def build_resnet50_aspp(pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, with_ibn, 
                    replace_stride_with_dilation=[False, True, True],
                    **kwargs)

@BACKBONE.register("R-DL-101-C1-C5")
def build_resnet101_aspp(pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, with_ibn, 
                    replace_stride_with_dilation=[False, True, True],
                    **kwargs)

@BACKBONE.register("R-DL-152-C1-C5")
def build_resnet152_aspp(pretrained=False, progress=True, with_ibn=False,**kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, with_ibn,  
                    replace_stride_with_dilation=[False, True, True],
                    **kwargs)

@BACKBONE.register("RX-DL-50-C1-C5")
def build_resnext50_32x4d_aspp(pretrained=False, progress=True, with_ibn=False,**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, with_ibn,  
                    replace_stride_with_dilation=[False, True, True],
                    **kwargs)

@BACKBONE.register("RX-DL-101-C1-C5")
def build_resnext101_32x8d_aspp(pretrained=False, progress=True, with_ibn=False,**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, with_ibn,  
                    replace_stride_with_dilation=[False, True, True],
                    **kwargs)


