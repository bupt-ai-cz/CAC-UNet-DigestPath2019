import torch
import torch.nn as nn
from ..ops.snconv import SNConv2d
import pdb

from ..registry import DISCRIMINATOR

class Discriminator(nn.Module):
    def __init__(self, in_channels, selected_layer, use_SN=False, fcn=False):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, kernal_size=3, pad=1, bn=True):
            if use_SN:
                conv = SNConv2d(in_filters, out_filters, kernal_size, 2, pad)
            else:
                conv = nn.Conv2d(in_filters, out_filters, kernal_size, 2, pad)
            block = [conv]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*block)
        self.channels = in_channels
        self.model = nn.ModuleList([
            discriminator_block(self.channels[selected_layer[0]], 32, 7, pad=3, bn=False),
            discriminator_block(self.channels[selected_layer[1]] + 32, 64, 3),
            discriminator_block(self.channels[selected_layer[2]] + 64, 128, 3),
            discriminator_block(self.channels[selected_layer[3]] + 128, 256, 3),
            discriminator_block(self.channels[selected_layer[4]] + 256, 512, 3),
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)
        self.fcn = fcn
        self.classifier = nn.Conv2d(512, 1, 3, 1, 1)
        self.selected_layer = selected_layer

    def forward(self, features):
        if not isinstance(features, list) and not isinstance(features, tuple):
            features = [features]
        input = features[self.selected_layer[0]]
        x = self.model[0](input)
        # pdb.set_trace()
        
        x = torch.cat((x, features[self.selected_layer[1]]), dim=1)
        x = self.model[1](x)

        # pdb.set_trace()
        x = torch.cat((x, features[self.selected_layer[2]]), dim=1)
        x = self.model[2](x)

        # pdb.set_trace()
        x = torch.cat((x, features[self.selected_layer[3]]), dim=1)
        x = self.model[3](x)

        # pdb.set_trace()
        x = torch.cat((x, features[self.selected_layer[4]]), dim=1)
        x = self.model[4](x)
                
        if self.fcn:
            return self.classifier(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, img):
        # if isinstance(img, list) or isinstance(img, tuple):
        #     img = img[-1]
        x = self.conv1(img)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


@DISCRIMINATOR.register("Decoder")
def base_discriminator(channels):
    selected_layer = [4, 3, 2, 1, 0]
    return Discriminator(channels, selected_layer)

@DISCRIMINATOR.register("Encoder")
def base_discriminator(channels):
    selected_layer = [0, 1, 2, 3, 4]
    return Discriminator(channels, selected_layer)


@DISCRIMINATOR.register("Semantic")
def base_discriminator(channels):
    return FCDiscriminator(channels[0])







