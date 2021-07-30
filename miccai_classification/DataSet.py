#
#author: Sachin Mehta
#Project Description: This repository contains source code for semantically segmenting WSIs; however, it could be easily
#                   adapted for other domains such as natural image segmentation
# File Description: This file is used to create data tuples
#==============================================================================

import cv2
import torch.utils.data
from PIL import Image

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imList, labelList, transform=None):
        self.imList = imList
        self.labelList = labelList
        self.transform = transform

    def __len__(self):
        return len(self.imList)
    # def __getitem__(self, index):
    #     """
    #     Args:
    #         index (int): Index

    #     Returns:
    #         tuple: (sample, target) where target is class_index of the target class.
    #     """
    #     path, target = self.samples[index]
    #     sample = self.loader(path)
    #     if self.transform is not None:
    #         sample = self.transform(sample)
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return sample, target
    def pil_loader(self, path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def __getitem__(self, idx):
        image_path = self.imList[idx]
        image = self.pil_loader(image_path)
        label_name = self.labelList[idx]
        # image = cv2.imread(image_name)
        # image = cv2.resize(image,(224,224))
        # label = cv2.imread(label_name, 0)
        # label = cv2.resize(label,(576,576))
        # _,label = cv2.threshold(label,127,1,cv2.THRESH_BINARY)
        if self.transform:
            image= self.transform(image)
        return image, label_name