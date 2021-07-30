import torch
import argparse
import os

from sseg.models.segmentors.generalized_segmentor import GeneralizedSegmentor
from sseg.models.segmentors.uda_segmentor import UDASegmentor
from sseg.models.default import cfg
# from sseg.models.backbones import resnet, efficientnet
from sseg.models.backbones import resnet
# from sseg.models.decoder import unet_decoder, unet_decoder_v2, deeplabv2_decoder, FPN_decoder, fcn_decoder
from sseg.models.decoder import unet_decoder_v2
from sseg.models.predictor import base_predictor
# from sseg.models.losses import bce_loss, dice_loss, mse_loss
from sseg.models.losses import bce_loss, dice_loss
# from sseg.models.discriminator import resnet_discriminator, base_discriminator
from sseg.models.discriminator import base_discriminator
from sseg.workflow.trainer import train_net

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch sseg")
    parser.add_argument(
        "--config_file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--resume_from")

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if cfg.MODEL.TYPE == "Generalized_Segmentor":
        net = GeneralizedSegmentor(cfg)
    elif cfg.MODEL.TYPE == "UDA_Segmentor" or cfg.MODEL.TYPE == "SDA_Segmentor":
        net = UDASegmentor(cfg)
    elif cfg.MODEL.TYPE == "UDA_Classifier":
        net = UDAClassifier(cfg)
    else:
        net = Classifier(cfg)
    torch.cuda.set_device(args.gpu_id)
    last_cp = os.path.join(cfg.WORK_DIR, 'last_epoch.pth')
    if args.resume_from:
        target_device = 'cuda:' + str(args.gpu_id)
        state_dict = torch.load(args.resume_from, map_location={
            'cuda:0': target_device,
            'cuda:1': target_device,
            'cuda:2': target_device})
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    elif os.path.exists(last_cp):
        # resume
        resume_info = os.path.join(cfg.WORK_DIR, 'resume')
        target_device = 'cuda:' + str(args.gpu_id)
        state_dict = torch.load(last_cp, map_location={
            'cuda:0': target_device,
            'cuda:1': target_device,
            'cuda:2': target_device})
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    net.cuda()

    train_net(net=net, cfg=cfg)
