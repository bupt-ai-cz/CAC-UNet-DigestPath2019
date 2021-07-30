import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.upsample import UnetUpsampleV2
from ..modules.basicblock import BasicResConvBlock
from ..ops.spp import PPM
from ..registry import DECODER

def pad(x1, x2):
    # padding 
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, (0, diffX, 0, diffY))
    return x1 
class UnetDecoderV2(nn.Module):
    '''
    UnetDecoderV2 
    1) Hypercolumn
    2) Center PPM
    3) IBN
    4) SCSE (attention module)
    5) dropout 
    '''
    def __init__(self, out_layers, input_channels, is_uda=False, replace_stride_with_dilation=False, use_ppm=False, use_attention=True):
        super(UnetDecoderV2,self).__init__()
        self.out_layers = out_layers
        self.is_uda = is_uda
        self.with_dilation = replace_stride_with_dilation
        self.use_ppm = use_ppm
        self.use_attention = use_attention

        self.center = nn.Sequential(
            BasicResConvBlock(input_channels[-1], input_channels[-1]//4),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        if self.use_ppm:
            self.ppm = nn.Sequential(
            PPM(input_channels[-1], input_channels[-1]//4),
            nn.Conv2d(input_channels[-1]*2, input_channels[-1], kernel_size=1),
            nn.BatchNorm2d(input_channels[-1]),
            nn.ReLU(inplace=True)
        ) 
            
        self.up0 = UnetUpsampleV2(input_channels[-1]//4, input_channels[-1], use_attention=self.use_attention)
        self.up1 = UnetUpsampleV2(input_channels[-1]//4, input_channels[-2], use_attention=self.use_attention) if not self.with_dilation \
                    else UnetUpsampleV2(input_channels[-1]//4, input_channels[-2], scale_factor=1, use_attention=self.use_attention)
        self.up2 = UnetUpsampleV2(input_channels[-2]//4, input_channels[-3], use_attention=self.use_attention) if not self.with_dilation \
                    else UnetUpsampleV2(input_channels[-2]//4, input_channels[-3], scale_factor=1, use_attention=self.use_attention)
        self.up3 = UnetUpsampleV2(input_channels[-3]//4, input_channels[-4], use_attention=self.use_attention)
        self.up4 = UnetUpsampleV2(input_channels[-4]//4, input_channels[-5], use_attention=self.use_attention) 
        self.upsamples = [self.up0, self.up1, self.up2, self.up3, self.up4]
        
        self.up_out_center = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True) if not self.with_dilation \
                        else nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_out_0 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True) if not self.with_dilation \
                        else nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_out_1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) if not self.with_dilation \
                        else nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_out_2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_out_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_outs = [self.up_out_0, self.up_out_1, self.up_out_2, self.up_out_3]

        self.out_conv_center = nn.Sequential(nn.Conv2d(input_channels[-1]//4, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.out_conv_0 = nn.Sequential(nn.Conv2d(input_channels[-1]//4, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.out_conv_1 = nn.Sequential(nn.Conv2d(input_channels[-2]//4, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.out_conv_2 = nn.Sequential(nn.Conv2d(input_channels[-3]//4, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.out_conv_3 = nn.Sequential(nn.Conv2d(input_channels[-4]//4, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.out_conv_4 = nn.Sequential(nn.Conv2d(input_channels[-5]//4, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.out_convs = [self.out_conv_0, self.out_conv_1, self.out_conv_2, self.out_conv_3, self.out_conv_4]

        self.dropout = nn.Dropout2d(p=0.3)

        self.out_channels = [64] * 5 + [64*6]
        self.out_channels = [self.out_channels[i] for i in self.out_layers]

    def forward(self, features):
        if len(features) != 5:
            raise ValueError(
                "Expected 5 features, got {} features instead.".format(
                    len(features)))

        outs = []
        if self.use_ppm:
            features = list(features)
            features[4] = self.ppm(features[4])
        center = self.center(features[4])
        decode_feature = center
        # outs.append(self.out_convs[0](decode_feature))
        for i in range(5):
            decode_feature = self.upsamples[i](decode_feature, features[4-i])
            out = self.out_convs[i](decode_feature)
            outs.append(out)

        outmaps = []
        for i in range(4):
            outmap = self.up_outs[i](outs[i])
            outmap = pad(outmap, outs[4])
            outmaps.append(outmap)
        outmaps.append(outs[4])            

        hypercolumn = torch.cat(outmaps + [pad(self.up_out_center(self.out_conv_center(center)), outs[4])],1)
        hypercolumn = self.dropout(hypercolumn)
        outmaps.append(hypercolumn)

        selected_outs = []
        selected_outs_without_up = []
        for i, out in enumerate(outmaps):
            if i in self.out_layers:
                selected_outs.append(out)
                if self.is_uda and i < 5:
                    selected_outs_without_up.append(outs[i])

        if self.is_uda:
            selected_outs_without_up.append(hypercolumn)
            return selected_outs, selected_outs_without_up
        
        return selected_outs

@DECODER.register("UnetDecoderV2-D5")
def build_unetdecoder_d5(input_channels):
    out_layers = [4]
    return UnetDecoderV2(out_layers, input_channels)

@DECODER.register("UnetDecoderV2-H")
def build_unetdecoder_d5(input_channels):
    out_layers = [5]
    return UnetDecoderV2(out_layers, input_channels)

@DECODER.register("UnetDecoderV2-D1-D5-H")
def build_unetdecoder_d1_d5(input_channels):
    out_layers = [0, 1, 2, 3, 4, 5]
    return UnetDecoderV2(out_layers, input_channels)

@DECODER.register("UnetDecoderV2-D5-UDA")
def build_unetdecoder_d1_d5_uda(input_channels):
    out_layers = [0, 1, 2, 3, 4]
    return UnetDecoderV2(out_layers, input_channels, is_uda=True)

@DECODER.register("UnetDecoderV2-D1-D5-UDA")
def build_unetdecoder_d1_d5_h_uda(input_channels):
    out_layers = [0, 1, 2, 3, 4, 5]
    return UnetDecoderV2(out_layers, input_channels, is_uda=True)


@DECODER.register("UnetDecoderV2-D5-PPM")
def build_unetdecoder_d5(input_channels):
    out_layers = [4]
    return UnetDecoderV2(out_layers, input_channels, use_ppm=True)

@DECODER.register("UnetDecoderV2-D1-D5-H-PPM")
def build_unetdecoder_d1_d5_PPM(input_channels):
    out_layers = [0, 1, 2, 3, 4, 5]
    return UnetDecoderV2(out_layers, input_channels, use_ppm=True)

@DECODER.register("UnetDecoderV2-H-PPM")
def build_unetdecoder_d5_PPM(input_channels):
    out_layers = [5]
    return UnetDecoderV2(out_layers, input_channels, use_ppm=True)

@DECODER.register("UnetDecoderV2-D1-D5-UDA-PPM")
def build_unetdecoder_d1_d5_uda_PPM(input_channels):
    out_layers = [0, 1, 2, 3, 4, 5]
    return UnetDecoderV2(out_layers, input_channels, is_uda=True, use_ppm=True)



@DECODER.register("UnetDecoderV2-NA-D5")
def build_unetdecoder_d5(input_channels):
    out_layers = [4]
    return UnetDecoderV2(out_layers, input_channels, use_attention=False)

@DECODER.register("UnetDecoderV2-NA-H")
def build_unetdecoder_d5(input_channels):
    out_layers = [5]
    return UnetDecoderV2(out_layers, input_channels, use_attention=False)

@DECODER.register("UnetDecoderV2-NA-D1-D5-H")
def build_unetdecoder_d1_d5(input_channels):
    out_layers = [0, 1, 2, 3, 4, 5]
    return UnetDecoderV2(out_layers, input_channels, use_attention=False)

@DECODER.register("UnetDecoderV2-NA-D1-D5-UDA")
def build_unetdecoder_d1_d5_uda(input_channels):
    out_layers = [0, 1, 2, 3, 4, 5]
    return UnetDecoderV2(out_layers, input_channels, is_uda=True, use_attention=False)
