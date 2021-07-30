import torch.nn as nn
import torch.nn.functional as F

from ..registry import PREDICTOR

@PREDICTOR.register("BasePredictor")
class BasePredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BasePredictor,self).__init__()
        if isinstance(in_channels, list):
            in_channels = in_channels[-1]
        # self.up = nn.ConvTranspose2d(in_channels, in_channels, 3, padding=1, stride=2)
        # self.conv1_1 = nn.Conv2d(in_channels, num_classes, 1)
        self.up = nn.ConvTranspose2d(in_channels, in_channels//4, 3, padding=1, stride=2)
        self.conv1_1 = nn.Conv2d(in_channels//4, num_classes, 1)
        self.out_channels = [num_classes]
        self.num_classes = num_classes

    def forward(self, input, target):
        if isinstance(input, list):
            input = input[-1]
        x = self.up(input)

        # input is CHW
        diffY = target.size()[2] - x.size()[2]
        diffX = target.size()[3] - x.size()[3]

        x = F.pad(x, (0, diffX, 0, diffY))

        return self.conv1_1(x) 



@PREDICTOR.register("UpsamplePredictor")
class UpsamplePredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UpsamplePredictor,self).__init__()
        if isinstance(in_channels, list):
            in_channels = in_channels[-1]
        self.conv1_1 = nn.Conv2d(in_channels, num_classes, 1)
        self.out_channels = [num_classes]
        self.num_classes = num_classes

    def forward(self, input, target):
        if isinstance(input, list):
            input = input[-1]
        input = self.conv1_1(input)
        x = F.interpolate(input, size=(target.size()[2], target.size()[3]))
        return x

