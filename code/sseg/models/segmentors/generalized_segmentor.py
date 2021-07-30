import torch.nn as nn
from ..backbones.backbone_builder import build_backbone
from ..decoder.decoder_builder import build_decoder
from ..predictor.predictor_builder import build_predictor
from ..losses.loss_builder import build_loss

class GeneralizedSegmentor(nn.Module):
    '''
    encoder + decoder
    1) FCN
    2) U-Net
    '''
    def __init__(self, cfg):
        super(GeneralizedSegmentor, self).__init__()

        self.backbone = build_backbone(cfg)
        self.decoder = build_decoder(cfg, self.backbone.out_channels)
        self.predictor = build_predictor(cfg, self.decoder.out_channels)
        self.loss = build_loss(cfg)
        self.cfg = cfg

    def forward(self, images, targets=None):
        features = self.backbone(images)
        decoder_out = self.decoder(features)
        logits = self.predictor(decoder_out, images)

        if self.training:
            losses = {}
            mask_loss = self.loss(logits, targets, weights=self.cfg.TRAIN.WEIGHT, pos_weights=self.cfg.TRAIN.POS_WEIGHT)
            loss = {"mask_loss": mask_loss}
            losses.update(loss)
            return losses
        
        return logits