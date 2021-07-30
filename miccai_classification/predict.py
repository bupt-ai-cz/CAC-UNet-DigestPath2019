import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms
import time
import datetime
import os
from PIL import Image, ImageOps
import sys
import torch.nn.functional as F
import numpy as np
import pdb
import argparse
import glob
import re
from tqdm import tqdm
import math
import cv2

Image.MAX_IMAGE_PIXELS = 933120000


def is_blank(source):
    source = np.array(source).astype('uint8')
    channels = cv2.split(source)
    return all([c.std() < 30 for c in channels])


def predict(args):
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # 这一句并没有用
    # images_path = {'neg': '../miccai/val_neg_wsi.txt', 'pos': '../miccai/val_pos_wsi.txt'}

    dir_path = {'neg': '../miccai/crop_neg', 'pos': '../miccai/tissue-train-pos'}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    state_dict = torch.load(args.modelpath)
    loaded_model = state_dict['model']
    net = torchvision.models.densenet161(num_classes=2)
    # net = torchvision.models.resnet101(num_classes=2)
    net.eval()
    net.load_state_dict(loaded_model)
    net.to(device)
    ##validate
    # pdb.set_trace()

    # 这几行没有用
    # all_wsi_neg = os.listdir('../tissue-train-neg/tissue-train-neg')
    # all_wsi_pos = os.listdir('../tissue-train-pos-v1/tissue-train-pos-v1')
    # all_wsi_pos = [wsi for wsi in all_wsi_pos if not 'mask' in wsi]

    txt_1 = open('../miccai/10fold/fold_8.txt')
    txt_2 = open('../miccai/10fold/fold_9.txt')
    images_path = txt_1.readlines() + txt_2.readlines()
    images_neg_path = [p.strip().split('/')[-1] for p in images_path if not 'pos' in p]
    images_pos_path = [p.strip().split('/')[-1] for p in images_path if 'pos' in p]
    # images_neg_path = set(all_wsi_neg) - set([p.strip().split('/')[-1] for p in images_path if not 'pos' in p])
    # images_pos_path = set(all_wsi_pos) - set([p.strip().split('/')[-1] for p in images_path if 'pos' in p])
    # print(len(all_wsi_neg),len(all_wsi_pos),len(images_neg_path),len(images_pos_path))
    images_neg_path = [os.path.join(dir_path['neg'], path.strip()) + ',0' for path in images_neg_path]
    images_pos_path = [os.path.join(dir_path['pos'], path.strip()) + ',1' for path in images_pos_path]
    txt_1.close()
    txt_2.close()

    result = np.zeros([2, 2])
    images_list = images_neg_path + images_pos_path
    wsi_labels = []
    wsi_scores = []
    IM_WIDTH = 1536
    IM_HEIGHT = 1536
    size = 1
    # record_5 = open('../miccai/10fold/record_5.txt','w')
    with torch.no_grad():
        for image_path in images_list:
            pre_labels = []
            pre_scores = []
            ground_truth = image_path.split(',')[-1]
            ## crop and predict
            # img_path = image_path.split(',')[0]
            # wsi_img = Image.open(img_path)
            # w,h = wsi_img.size
            # cols = int(math.ceil(w / IM_WIDTH))
            # rows = int(math.ceil(h / IM_HEIGHT))
            # _right = cols * IM_WIDTH - w
            # _bottom = rows * IM_HEIGHT - h
            # img = ImageOps.expand(wsi_img,(0,0,_right,_bottom),fill='white')
            # for i in range(rows):
            #     for j in range(cols):
            #         patch = img.crop((j*IM_WIDTH, i*IM_HEIGHT, int((j+size)*IM_WIDTH), int((i+size)*IM_HEIGHT))).convert('RGB')
            #         if not is_blank(patch):
            #             imgblob = test_transforms(patch).unsqueeze(0).to(device)
            #             imgblob = Variable(imgblob)
            #             probability = F.softmax(net(imgblob),dim=1)
            #             _,label = torch.max(probability,1)
            #             pre_labels.append(label)
            #             pre_scores.append(probability[0,1])

            wsi = os.path.splitext(image_path.split(',')[0])[0]
            # 获得所有以wsi所存储的文件名开头的文件列表，表示一张大图分割成的所有小图
            patchs = glob.glob(wsi + '*')
            print(len(patchs))

            # 预测一张大图内的所有小图的类别
            for patch in patchs:
                image = Image.open(patch)
                imgblob = test_transforms(image).unsqueeze(0).to(device)
                imgblob = Variable(imgblob)
                probability = F.softmax(net(imgblob), dim=1)
                # pdb.set_trace()
                _, label = torch.max(probability, 1)
                pre_labels.append(label)
                pre_scores.append(probability[0, 1])

            # 综合所有小图的预测类别来获得大图的预测类别
            if np.sum(np.array(pre_labels)) / float(len(pre_labels)) > 0.1:
                wsi_pre_label = 1
                wsi_pre_score = np.mean(np.array(pre_scores)[np.where(np.array(pre_labels) == 1)])
                # print(wsi_pre_score,image_path)
            else:
                wsi_pre_label = 0
                wsi_pre_score = np.mean(np.array(pre_scores)[np.where(np.array(pre_labels) == 0)])
            # if wsi_pre_label-int(ground_truth) != 0 or (int(ground_truth)==0 and wsi_pre_score >0.2) or (int(ground_truth)==1 and wsi_pre_score<0.8):
            #     print(img_path,wsi_pre_label,ground_truth)
            #     record_5.write(img_path+'\n')
            wsi_labels.append(int(ground_truth))
            wsi_scores.append(wsi_pre_score)
            print('ground_truth is', ground_truth, wsi_pre_label)
            result[int(ground_truth)][int(wsi_pre_label)] = result[int(ground_truth)][int(wsi_pre_label)] + 1
    roc = roc_auc_score(np.array(wsi_labels), np.array(wsi_scores))
    # record_5.close()
    print('result {}'.format(result))
    print('roc {}'.format(roc))


def predict_chaoyang(args):
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    state_dict = torch.load(args.modelpath)
    loaded_model = state_dict['model']
    # net = torchvision.models.densenet161(num_classes=2)
    net = torchvision.models.resnet101(num_classes=2)
    net.eval()
    net.load_state_dict(loaded_model)
    net.to(device)
    result = np.zeros([2, 2])
    wsi_list = os.listdir('../chaoyang')
    wsi_labels = []
    wsi_scores = []
    with torch.no_grad():
        for wsi in tqdm(wsi_list):
            pre_labels = []
            pre_scores = []
            ground_truth = '1'
            # wsi = os.path.splitext(image_path.split(',')[0])[0]
            patchs = glob.glob('../chaoyang/' + wsi + '/*.JPG')
            print(len(patchs))
            for patch in patchs:
                image = Image.open(patch)
                imgblob = test_transforms(image).unsqueeze(0).to(device)
                imgblob = Variable(imgblob)
                # imgblob = imgblob.to(device)
                probability = F.softmax(net(imgblob), dim=1)
                _, label = torch.max(probability, 1)
                pre_labels.append(label)
                pre_scores.append(probability[0, 1])
            if np.sum(np.array(pre_labels)) / float(len(pre_labels)) > 0.1:
                wsi_pre_label = 1
                wsi_pre_score = np.mean(np.array(pre_scores)[np.where(np.array(pre_labels) == 1)])
                print(wsi_pre_score)
            else:
                wsi_pre_label = 0
                wsi_pre_score = np.mean(np.array(pre_scores)[np.where(np.array(pre_labels) == 0)])
            wsi_labels.append(wsi_pre_label)
            wsi_scores.append(wsi_pre_score)
            print('ground_truth is', ground_truth, wsi_pre_label)
            result[int(ground_truth)][int(wsi_pre_label)] = result[int(ground_truth)][int(wsi_pre_label)] + 1
    # roc = roc_auc_score(np.array(wsi_labels),np.array(wsi_scores))
    print('result {}'.format(result))
    # print('roc {}'.format(roc))


def parse_args():
    pathes = {'res': 'result/res/model_3.pth',
              'dense': 'result/dense/5_retrain/model_4.pth',
              'googlenet': 'result/googlenet/model_7.pth'}
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="densenet161", help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--modelpath', default=pathes['dense'], help='model path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    predict(args)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # predict_chaoyang(args)
