import torch
import torch.nn as nn
import torch.nn.functional as F
from .basicblock import BasicResConvBlock
from ..ops.scse import SCSEBlock


'''
Unet model module 
'''
class UnetUpsample(nn.Module):
    def __init__(self, inchannels, channels, scale_factor=2):
        super(UnetUpsample, self).__init__()
        self.up = nn.ConvTranspose2d(inchannels, channels, 3, padding=1, stride=scale_factor)
        self.conv = BasicResConvBlock(channels + channels, channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
    
        # padding 
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (0, diffX, 0, diffY))

        x = torch.cat([x2, x1], dim=1)
        
        out = self.conv(x)
        return out

'''
Unet model module 
'''
class UnetUpsampleV2(nn.Module):
    def __init__(self, inchannels, channels, scale_factor=2, bilinear=False, use_attention=True):
        super(UnetUpsampleV2, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(inchannels, inchannels, 3, padding=1, stride=scale_factor),
                nn.BatchNorm2d(inchannels),
                nn.ReLU(inplace=True),
            )
        self.scse = SCSEBlock(channels + inchannels)
        self.conv = BasicResConvBlock(channels + inchannels, channels//4, ibn=True, dropout=False)
        self.use_attention = use_attention

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # padding 
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (0, diffX, 0, diffY))

        x = torch.cat([x2, x1], dim=1)
        
        if self.use_attention:
            x = self.scse(x)
        out = self.conv(x)
        return out
    
def pad(x1, x2):
    # padding 
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, (0, diffX, 0, diffY))
    return x1 