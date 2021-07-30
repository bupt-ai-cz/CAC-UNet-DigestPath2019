import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from ..datasets.loader.dataset import BaseDataset
# from ..datasets.loader.gtav_dataset import GTAVDataset
# from ..datasets.loader.cityscapes_dataset import CityscapesDataset
from ..datasets.loader.digest_dataset import DigestDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler, RandomSampler
from torch.autograd import Variable
from ..datasets.metrics.miou import mean_iou, get_hist
from ..datasets.metrics.acc import acc, acc_with_hist
from ..models.losses.ranger import Ranger
from ..models.losses.cos_annealing_with_restart import CosineAnnealingLR_with_Restart
from ..models.registry import DATASET

import os
import logging
import time
import numpy as np
import random
import pickle
import pdb

def seed_everything(seed=1234):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

def train_net(net,
              cfg,
              ):
    seed_everything()
    
    dir_cp = cfg.WORK_DIR
    if(not os.path.exists(dir_cp)):
        os.makedirs(dir_cp)
    logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                    filename=os.path.join(cfg.WORK_DIR,'train.log'),
                    filemode='a',
                    level=logging.INFO)
    logger = logging.getLogger("sseg.trainer")
    sh = logging.StreamHandler()
    logger.addHandler(sh)

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


    # train dataset
    train = DATASET[cfg.DATASET.TYPE](anns, image_dir, scale=scale, center_crop=train_center_crop, resize_size=train_resize_size, use_aug=use_aug)
    if cfg.DATASET.NUM_SAMPLES > 0:
        weights = [1.0] * len(train)
        sample_len = cfg.DATASET.NUM_SAMPLES
        weighted_sampler = WeightedRandomSampler(weights, sample_len, replacement=True)
        train_data = DataLoader(train, cfg.TRAIN.BATCHSIZE, num_workers=4, drop_last=True, sampler=weighted_sampler, pin_memory=True)
    else:
        train_data = DataLoader(train, cfg.TRAIN.BATCHSIZE, shuffle=True, num_workers=4, drop_last=True)

    # target dataset
    if cfg.DATASET.TARGET.ANNS != '':
        t_anns = cfg.DATASET.TARGET.ANNS
        t_image_dir = cfg.DATASET.TARGET.IMAGEDIR
        t_train = DATASET[cfg.DATASET.TARGET.TYPE](t_anns, t_image_dir, scale=scale, center_crop=train_center_crop, resize_size=train_resize_size, use_aug=use_aug)
        t_train_data = DataLoader(t_train, cfg.TRAIN.BATCHSIZE, shuffle=True, num_workers=4, drop_last=True)
        target_iter = iter(t_train_data)
    
    # val dataset  
    val_anns = cfg.DATASET.VAL.ANNS
    val_image_dir = cfg.DATASET.VAL.IMAGEDIR
    val = DATASET[cfg.DATASET.VAL.TYPE](val_anns, val_image_dir, scale=scale, center_crop=val_center_crop, resize_size=val_resize_size, )
    val_data = DataLoader(val, 4, num_workers=4, drop_last=True, shuffle=True)

    n_train = cfg.DATASET.NUM_SAMPLES if cfg.DATASET.NUM_SAMPLES>0 else len(train) 
    expect_iter = n_train * cfg.TRAIN.EPOCHES //cfg.TRAIN.BATCHSIZE

    # optimizer 
    optimizer, D_optimizer_dict = build_optimizer(net, cfg)
    # pdb.set_trace()
    scheduler = None
    if cfg.TRAIN.SCHEDULER == "CosineAnnealingLR_with_Restart":
        scheduler = CosineAnnealingLR_with_Restart(
            optimizer, 
            T_max=cfg.TRAIN.COSINEANNEALINGLR.T_MAX*expect_iter//cfg.TRAIN.EPOCHES, 
            T_mult=cfg.TRAIN.COSINEANNEALINGLR.T_MULT)
    elif cfg.TRAIN.SCHEDULER == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.MULTISTEPLR.MILESTONES, gamma=cfg.TRAIN.MULTISTEPLR.GAMMA)
    

    # resume
    resume_info = os.path.join(cfg.WORK_DIR, 'resume')
    result_info = os.path.join(cfg.WORK_DIR, 'result')
    if os.path.exists(resume_info):
        with open(resume_info, 'r') as f:
            last_epoch = int(f.read())
    else:
        last_epoch = 0
    if scheduler:
        scheduler.last_epoch = last_epoch
    if os.path.exists(result_info):
        with open(result_info, 'rb') as f:
            result = pickle.loads(f.read())
    
    logger.info(cfg)
    logger.info("resume from epoch %d"%last_epoch )
    logger.info("Start training!")

    # apex
    from apex import amp
    optimizers = [optimizer, ]
    for name, optim in D_optimizer_dict.items():
        optimizers.append(optim)
    net, optimizers = amp.initialize(net, optimizers, opt_level=cfg.TRAIN.APEX_OPT)
    for i, (name, optim) in enumerate(D_optimizer_dict.items()):
        # net, optim = amp.initialize(net, optim, opt_level=cfg.TRAIN.APEX_OPT)
        D_optimizer_dict[name] = optimizers[i+1]

    max_metrics = 0
    max_metrics_epoch = 0
    metrics_decay_count = 0
    for epoch in range(last_epoch, cfg.TRAIN.EPOCHES):
        net.train()
        epoch_total_loss = {}
        epoch_total_loss['loss'] = 0
        iter_cnt = 0
        
        for i, b in enumerate(train_data):
            start = time.time()
            # pdb.set_trace()

            images = Variable(b[0].cuda())
            labels = Variable(b[1].type(torch.LongTensor).cuda())
            # has_target = Variable(b[2].cuda()) 
            # classify_label = Variable(b[3].cuda()) 

            # uda target dataset
            if cfg.DATASET.TARGET.ANNS != '':
                try:
                    t = next(target_iter)
                except StopIteration:
                    target_iter = iter(DataLoader(t_train, cfg.TRAIN.BATCHSIZE, shuffle=True, num_workers=4, drop_last=True))
                    t = next(target_iter)
                t_images = Variable(t[0].cuda())
                t_labels = Variable(t[1].type(torch.LongTensor).cuda())

            if cfg.MODEL.TYPE == "UDA_Segmentor":
                loss_dict = net(
                    source=images,
                    target=t_images,
                    source_label=labels,
                    target_label=t_labels,
                    )
            else:
                loss_dict = net(images, labels) 

            if len(loss_dict) > 1:
                for loss_name, loss_value in loss_dict.items():
                    epoch_total_loss[loss_name] = loss_value.item() if i == 0 else epoch_total_loss[loss_name] + loss_value.item()       

            loss = sum(loss for name, loss in loss_dict.items() if "D_" not in name)
                
            epoch_total_loss['loss'] += loss.item()

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            for name, optim in D_optimizer_dict.items():
                name = 'D_' + name + '_loss'
                if iter_cnt % cfg.MODEL.DISCRIMINATOR.UPDATE_T == 0:
                    optim.zero_grad()
                    with amp.scale_loss(loss_dict[name], optim) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                    optim.step()
                    
            if cfg.TRAIN.SCHEDULER == "CosineAnnealingLR_with_Restart":
                scheduler.step()

            iter_cnt += 1
            end = time.time()
            if iter_cnt%cfg.TRAIN.ITER_REPORT ==0:
                eta = itv2time((end - start) * (expect_iter - epoch*n_train//cfg.TRAIN.BATCHSIZE - iter_cnt))
                logger.info('eta: {}, epoch: {}, iter: {:.2f}%, lr: {:.2e}'
                            .format(eta, epoch + 1, 100*iter_cnt*cfg.TRAIN.BATCHSIZE/n_train, optimizer.param_groups[-1]['lr']) + print_loss_dict(epoch_total_loss, iter_cnt))
        
        if cfg.TRAIN.SCHEDULER == "MultiStepLR":
            scheduler.step()

        if cfg.DATASET.VAL.ANNS != '' and (epoch + 1) % cfg.TRAIN.ITER_VAL == 0:
            with torch.no_grad():
                net.eval()
                n_class = cfg.MODEL.PREDICTOR.NUM_CLASSES
                hist = np.zeros((n_class, n_class))
                iu_list = []
                dice_list = []
                for i, b in enumerate(val_data):
                    images = Variable(b[0].cuda())
                    labels = Variable(b[1].type(torch.LongTensor).cuda())
                    logits = net(images)

                    label_pred = logits.max(dim=1)[1].data.cpu().numpy()
                    label_true = labels.data.cpu().numpy()

                    if cfg.DATASET.VAL.REDUCE == 'mean':
                        for i in range(4):
                            item_hist =get_hist(label_true[i], label_pred[i], n_class) 
                            item_iu = (np.diag(item_hist) + 0.00001)/ (item_hist.sum(axis=1) + item_hist.sum(axis=0) - np.diag(item_hist) + 0.00001)
                            item_dice = (np.diag(item_hist) * 2 + 0.00001)/ (item_hist.sum(axis=1) + item_hist.sum(axis=0) + 0.00001)
                            iu_list.append(item_iu)
                            dice_list.append(item_dice)
                    else:
                        item_hist =get_hist(label_true, label_pred, n_class)
                        hist += item_hist

                if cfg.DATASET.VAL.REDUCE == 'mean':
                    iu = sum(iu_list)/len(iu_list)
                    dice = sum(dice_list)/len(iu_list)
                else:
                    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 0.00001)
                    dice = np.diag(hist) * 2/ (hist.sum(axis=1) + hist.sum(axis=0) + 0.00001)
                mean_iu = np.nanmean(iu)
                mean_dice = np.nanmean(dice[1:])

                result_item = {'epoch': epoch+1}
                result_item.update({'iou': mean_iu})
                result_item.update(result_list2dict(iu,'iou'))
                result_item.update({'dice': mean_dice})
                result_item.update(result_list2dict(dice,'dice'))
                result.append(result_item)
                with open(result_info, 'wb') as f:
                    f.write(pickle.dumps(result))
                # logger.info('epoch: {}, val_miou: {:.4f}({:.4f})'.format(epoch + 1, mean_iu, print_top(result, 'iou')) + print_iou_list(iu))
                logger.info('epoch: {}, val_dice: {:.4f}({:.4f})'.format(epoch + 1, mean_dice, print_top(result, 'dice')) + print_iou_list(dice))
                
                # early stopping
                if mean_dice >= max_metrics:
                    max_metrics = mean_dice
                    max_metrics_epoch = epoch + 1
                    metrics_decay_count = 0
                else:
                    metrics_decay_count += 1
                if metrics_decay_count > early_stopping and early_stopping >= 0:
                    logger.info('early stopping! epoch{} max metrics: {:.4f}'.format(max_metrics_epoch, max_metrics))
                    break

        if cfg.TRAIN.SAVE_ALL:
            torch.save(net.state_dict(), os.path.join(dir_cp, 'CP{}.pth'.format(epoch + 1)))
            torch.save(net.state_dict(), os.path.join(dir_cp, 'last_epoch.pth'))
            if max_metrics_epoch == epoch + 1:
                torch.save(net.state_dict(), os.path.join(dir_cp, 'val_highest.pth'))
        else:
            torch.save(net.state_dict(), os.path.join(dir_cp, 'last_epoch.pth'))
            if max_metrics_epoch == epoch + 1:
                torch.save(net.state_dict(), os.path.join(dir_cp, 'val_highest.pth'))
                # torch.save(net, os.path.join(dir_cp, 'val_highest_with_model.pth'))

        with open(resume_info, 'w') as f:
            f.write(str(epoch+1))

        
def build_optimizer(model, cfg):
    optimizer = cfg.TRAIN.OPTIMIZER
    lr = cfg.TRAIN.LR
    param = [
        {'params': model.backbone.parameters(), "lr": lr*0.1},
        {'params': model.decoder.parameters(), "lr": lr},
        {'params': model.predictor.parameters(), "lr": lr}
        ]
    if(optimizer == 'SGD'):
        optimizer = optim.SGD(param, momentum=0.9, weight_decay=0.0005)
    elif(optimizer == 'Adam'):
        optimizer = optim.Adam(param, betas=(0.9, 0.999), weight_decay=0.0005)
    elif(optimizer == 'Ranger'):
        optimizer = Ranger(param, weight_decay=0.0005)
    else:
        optimizer = optim.SGD(param, momentum=0.9, weight_decay=0.0005)

    D_optimizer_dict = {}
    if len(cfg.MODEL.DISCRIMINATOR.TYPE)>0:
        d_params = {}
        for d_name, D in model.discriminators.named_children():
            d_params[d_name] = D.parameters()

        for i, d_name in enumerate(cfg.MODEL.DISCRIMINATOR.TYPE):
            D_optimizer_dict[d_name] = optim.Adam(model.discriminators.parameters(), lr=cfg.MODEL.DISCRIMINATOR.LR[i], betas=(0.9, 0.999))
        
    return optimizer, D_optimizer_dict


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

def print_top(result, metrics, top=0.1):
    res = np.array([x[metrics] for x in result])
    res = np.sort(res)
    top = int(len(res) * 0.1) + 1
    return res[-top:].mean()

def result_list2dict(iou_list, metrics):
    res = {}
    for i, iou in enumerate(iou_list):
        res[metrics+str(i)] = iou
    return res

def itv2time(iItv):
    h = int(iItv//3600)
    sUp_h = iItv-3600*h
    m = int(sUp_h//60)
    sUp_m = sUp_h-60*m
    s = int(sUp_m)
    return "{}h {:0>2d}min".format(h,m,s)