import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import pdb

from ..backbones.backbone_builder import build_backbone
from ..decoder.decoder_builder import build_decoder
from ..predictor.predictor_builder import build_predictor
from ..losses.loss_builder import build_loss
from ..discriminator.discriminator_builder import build_discriminator

class UDASegmentor(nn.Module):
    """
    unsupervised domain adaptation segmentor
    """
    def __init__(self, cfg):
        super(UDASegmentor, self).__init__()

        self.backbone = build_backbone(cfg)
        self.decoder = build_decoder(cfg, self.backbone.out_channels)
        self.predictor = build_predictor(cfg, self.decoder.out_channels)
        self.loss = build_loss(cfg)
        self.cfg = cfg

        self.discriminators = nn.Sequential()
        for name, D in build_discriminator(
            cfg, 
            self.backbone.out_channels, 
            self.decoder.out_channels, 
            self.predictor.out_channels
            ).items():
            self.discriminators.add_module(name, D)
        
        # todo
        # add discriminator loss to config file
        self.discriminator_loss = build_loss(cfg, is_discriminator=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, source, target=None, source_label=None, target_label=None):
        # source domain
        s_features = self.backbone(source)
        s_decoder_out = self.decoder(s_features)
        if isinstance(s_decoder_out, tuple):
            s_decoder_fm = s_decoder_out[1]
            s_decoder_out = s_decoder_out[0]
        else:
            s_decoder_fm = s_decoder_out
        if not isinstance(s_decoder_fm, list):
            s_decoder_fm = [s_decoder_fm]
        s_logits = self.predictor(s_decoder_out, source)

        if self.training:
            # target domain
            t_features = self.backbone(target)
            t_decoder_out = self.decoder(t_features)
            if isinstance(t_decoder_out, tuple):
                t_decoder_fm = t_decoder_out[1]
                t_decoder_out = t_decoder_out[0]
            else:
                t_decoder_fm = t_decoder_out
            if not isinstance(t_decoder_fm, list):
                t_decoder_fm = [t_decoder_fm]
            t_logits = self.predictor(t_decoder_out, target)
            t_logits_softmax = self.softmax(t_logits)
            s_logits_softmax = self.softmax(s_logits)

            # gt mask to onehot
            s_gt_onehot = make_one_hot(source_label, self.predictor.num_classes)
            t_gt_onehot = make_one_hot(target_label, self.predictor.num_classes)

            # update discriminators
            losses = {}
            for name, D in self.discriminators.named_children():
                # pdb.set_trace()
                if "Decoder" in name:
                    s_D_logits = D([x.detach() for x in s_decoder_fm])
                    t_D_logits = D([x.detach() for x in t_decoder_fm])
                elif "Encoder" in name:
                    s_D_logits = D([x.detach() for x in s_features])
                    t_D_logits = D([x.detach() for x in t_features])
                elif "Predictor" in name:
                    s_D_logits = D(s_logits.detach())
                    t_D_logits = D(t_logits.detach())
                elif "Semantic" in name:
                    s_D_logits = D(s_logits_softmax.detach())
                    t_D_logits = D(t_logits_softmax.detach())
                    s_gt_D_logits = D(s_gt_onehot)
                    t_gt_D_logits = D(t_gt_onehot)
                
                if "Semantic" not in name:
                    is_source = torch.zeros_like(s_D_logits).cuda()
                    is_target = torch.ones_like(s_D_logits).cuda()
                    if self.cfg.MODEL.DISCRIMINATOR.SMOOTH:
                        is_source = torch.ones_like(s_D_logits).cuda()
                        is_source = is_source*0.2
                        is_target = torch.ones_like(s_D_logits).cuda()
                        is_target = is_target*0.8
                    discriminator_loss = (self.discriminator_loss(s_D_logits, is_source) + 
                                      self.discriminator_loss(t_D_logits, is_target))/2
                else:
                    is_t_predict = torch.zeros_like(t_D_logits).cuda()
                    is_gt = torch.ones_like(t_D_logits).cuda()
                    discriminator_loss = (self.discriminator_loss(t_D_logits, is_t_predict) + 
                                        self.discriminator_loss(t_gt_D_logits, is_gt) + 
                                        self.discriminator_loss(s_D_logits, is_t_predict) + 
                                        self.discriminator_loss(s_gt_D_logits, is_gt) )/4
                losses.update({'D_' + name + '_loss': discriminator_loss})

            # adv_losses
            adv_losses = []
            for i, value in enumerate(self.discriminators.named_children()):
                name, D = value
                if "Decoder" in name:
                    s_D_logits = D(s_decoder_fm)
                    t_D_logits = D(t_decoder_fm)
                elif "Encoder" in name:
                    s_D_logits = D(s_features)
                    t_D_logits = D(t_features)
                elif "Predictor" in name:
                    s_D_logits = D(s_logits)
                    t_D_logits = D(t_logits)
                elif "Semantic" in name:
                    s_D_logits = D(s_logits_softmax)
                    t_D_logits = D(t_logits_softmax)
                    s_gt_D_logits = D(s_gt_onehot)
                    t_gt_D_logits = D(t_gt_onehot)

                if "Semantic" not in name:
                    is_source = torch.zeros_like(s_D_logits).cuda()
                    is_target = torch.ones_like(s_D_logits).cuda()
                    if self.cfg.MODEL.DISCRIMINATOR.SMOOTH:
                        is_source = torch.ones_like(s_D_logits).cuda()
                        is_source = is_source*0.2
                        is_target = torch.ones_like(s_D_logits).cuda()
                        is_target = is_target*0.8
                    discriminator_loss = self.discriminator_loss(t_D_logits, is_source) + self.discriminator_loss(t_D_logits, is_target)
                    adv_losses.append(self.cfg.MODEL.DISCRIMINATOR.WEIGHT[i] * discriminator_loss)
                else:
                    is_t_predict = torch.zeros_like(t_D_logits).cuda()
                    is_gt = torch.ones_like(t_D_logits).cuda()
                    discriminator_loss = self.discriminator_loss(t_D_logits, is_gt) + self.discriminator_loss(s_D_logits, is_gt)
                    adv_losses.append(self.cfg.MODEL.DISCRIMINATOR.WEIGHT[i] * discriminator_loss)
            
            # update seg loss
            # logits HyperCol
            mask_loss = self.loss(s_logits, source_label) + self.loss(t_logits, target_label) + sum(adv_losses)
           
            loss = {"mask_loss": mask_loss}
            losses.update(loss)
            return losses
        
        return s_logits



def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    input = input.unsqueeze(1)
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    result = result.scatter_(1, input, 1)

    return result
