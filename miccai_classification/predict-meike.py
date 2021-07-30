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
import csv

Image.MAX_IMAGE_PIXELS = 933120000


class classify:
    def __init__(self, model, model_path, img_path, threshold):
        self.model = model
        self.img_path = img_path
        self.model_path = model_path
        self.threshold = threshold
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.test_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            self.normalize,
        ])

    def is_blank(self, source):
        source = np.array(source).astype('uint8')
        channels = cv2.split(source)
        return all([c.std() < 30 for c in channels])

    def load_net(self, model_path):
        state_dict = torch.load(model_path)
        loaded_model = state_dict['model']
        if self.model == 'resnet50':
            net = torchvision.models.resnet50(num_classes=2)
        if self.model == 'densenet161':
            net = torchvision.models.densenet161(num_classes=2)
        if self.model == 'resnet101':
            net = torchvision.models.resnet101(num_classes=2)
        net.eval()
        net.load_state_dict(loaded_model)
        net.to('cuda')
        return net

    def predict(self, patch):
        net = self.load_net(self.model_path)
        imgblob = self.test_transforms(patch).unsqueeze(0).to("cuda")
        imgblob = Variable(imgblob)
        probability = F.softmax(net(imgblob), dim=1)
        _, label = torch.max(probability, 1)
        return label, probability[0, 1]

    def compute_wsi(self, pre_labels, pre_scores):
        if np.sum(np.array(pre_labels) / float(len(pre_labels))) > self.threshold:
            wsi_pre_label = 1
            wsi_pre_score = np.mean(np.array(pre_scores)[np.where(np.array(pre_labels) == 1)])
        else:
            wsi_pre_label = 0
            wsi_pre_score = np.mean(np.array(pre_scores)[np.where(np.array(pre_labels) == 0)])
        return wsi_pre_label, wsi_pre_score

    def run(self):
        IM_WIDTH = 1024
        IM_HEIGHT = 1024
        size = 1.5
        record = open('../output/predict.csv', 'a')
        writer = record.writer(record)
        # 预测每张大图的类别
        for wsi in self.img_path:
            pre_labels = []
            pre_scores = []
            img = Image.open(wsi)
            w, h = img.size
            cols = int(math.ceil((w - 1536) / IM_WIDTH))
            rows = int(math.ceil((h - 1536) / IM_HEIGHT))
            _right = cols * IM_WIDTH + 1536 - w
            _bottom = rows * IM_HEIGHT + 1536 - h
            img = ImageOps.expand(img, (0, 0, _right, _bottom), fill='white')
            for i in tqdm(range(rows + 1)):
                for j in range(cols + 1):
                    # 将一张大图切成很多小图
                    patch = img.crop((j * IM_WIDTH, i * IM_HEIGHT, int((j + size) * IM_WIDTH), int((i + size) * IM_HEIGHT))).convert('RGB')
                    if not self.is_blank(patch):
                        label, score = self.predict(patch)
                        pre_labels.append(label)
                        pre_scores.append(score)
            wsi_pre_label, wsi_pre_score = self.compute_wsi(pre_labels, pre_scores)
            writer.writerows([wsi, wsi_pre_label, wsi_pre_score])
        record.close()
