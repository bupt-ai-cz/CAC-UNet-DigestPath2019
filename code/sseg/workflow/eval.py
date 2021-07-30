import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from ..datasets.loader.dataset import BaseDataset
from ..datasets.loader.gtav_dataset import GTAVDataset
from ..datasets.loader.cityscapes_dataset import CityscapesDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler, RandomSampler
from torch.autograd import Variable
from ..datasets.metrics.miou import mean_iou, get_hist
from ..datasets.metrics.acc import acc, acc_with_hist
from ..models.losses.ranger import Ranger
from ..models.losses.cos_annealing_with_restart import CosineAnnealingLR_with_Restart
from ..models.registry import DATASET

import os
import time
import numpy as np
import tqdm
import pdb

def eval_net(net,
              cfg,
              ):
    
    # train dataset
    result = []
    early_stopping = cfg.TRAIN.EARLY_STOPPING
    anns = cfg.DATASET.ANNS
    image_dir = cfg.DATASET.IMAGEDIR
    with_empty = cfg.DATASET.WITH_EMPTY
    scale = cfg.DATASET.RESIZE_SCALE
    train_resize_size = cfg.DATASET.RESIZE_SIZE
    train_center_crop = cfg.DATASET.CENTER_CROP
    val_center_crop = cfg.DATASET.VAL.CENTER_CROP
    val_resize_size = cfg.DATASET.VAL.RESIZE_SIZE
    use_aug = cfg.DATASET.USE_AUG
    bs = 8

    
    # val dataset  
    val_anns = cfg.DATASET.VAL.ANNS
    val_image_dir = cfg.DATASET.VAL.IMAGEDIR
    val = DATASET[cfg.DATASET.VAL.TYPE](val_anns, val_image_dir, scale=scale, center_crop=val_center_crop, resize_size=val_resize_size)
    val_data = DataLoader(val, bs, num_workers=4, drop_last=True, shuffle=True)

    
    with torch.no_grad():
        if cfg.MODEL.TYPE == 'Generalized_Segmentor' or cfg.MODEL.TYPE == 'UDA_Segmentor':
            net.eval()
            n_class = cfg.MODEL.PREDICTOR.NUM_CLASSES
            hist = np.zeros((n_class, n_class))
            iu_list = []
            dice_list = []
            for i, b in enumerate(tqdm.tqdm(val_data)):
                images = Variable(b[0].cuda())
                labels = Variable(b[1].type(torch.LongTensor).cuda())

                logits = net(images)

                if isinstance(logits, tuple):
                    logits = logits[0]

                if isinstance(logits, list):
                    # logits = torch.cat(logits,0).max(dim=0)[0].unsqueeze(0)
                    logits = logits[-1]

                label_pred = logits.max(dim=1)[1].data.cpu().numpy()
                label_true = labels.data.cpu().numpy()

                item_hist =get_hist(label_true, label_pred, n_class)
                hist += item_hist
                if cfg.DATASET.VAL.REDUCE == 'mean':
                    item_iu = (np.diag(item_hist) + 0.00001)/ (item_hist.sum(axis=1) + item_hist.sum(axis=0) - np.diag(item_hist) + 0.00001)
                    item_dice = (np.diag(item_hist) * 2 + 0.00001)/ (item_hist.sum(axis=1) + item_hist.sum(axis=0) + 0.00001)
                    iu_list.append(item_iu)
                    dice_list.append(item_dice)
                    # pdb.set_trace()

            if cfg.DATASET.VAL.REDUCE == 'mean':
                iu = sum(iu_list)/len(iu_list)
                dice = sum(dice_list)/len(iu_list)
            else:
                iu = (np.diag(hist) + 0.00001)/ (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 0.00001)
                dice = (np.diag(hist) * 2 + 0.00001)/ (hist.sum(axis=1) + hist.sum(axis=0) + 0.00001)
            mean_iou_weight = sum(iu) / n_class
            # print('val_miou: {:.4f}'.format(mean_iou_weight) + print_iou_list(iu))
            print('val_dice: {:.4f}'.format(np.nanmean(dice[1:])) + print_iou_list(dice))

    


def print_loss_dict(loss_dict, iter_cnt):
    res = ''
    for loss_name, loss_value in loss_dict.items():
        res += ', {}: {:.6f}'.format(loss_name, loss_value/iter_cnt)
    return res

def print_iou_list(iou_list):
    res = ''
    for i, iou in enumerate(iou_list):
        res += ', {}: {:.4f}'.format(i, iou)
    return res


def itv2time(iItv):
    h = int(iItv//3600)
    sUp_h = iItv-3600*h
    m = int(sUp_h//60)
    sUp_m = sUp_h-60*m
    s = int(sUp_m)
    return "{}h {:0>2d}min".format(h,m,s)