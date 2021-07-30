import os
import glob
import json

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def get_path(image_dir, label_dir, image_suffix, label_suffix):
    '''
    read images and labels, return image_list and label_list (file path)
    '''
    all_file = os.listdir(image_dir)
    basic_names = [os.path.split(im)[-1].replace(image_suffix, '') for im in all_file 
                if image_suffix in os.path.split(im)[-1]]
    datas = [os.path.join(image_dir, id + image_suffix) for id in basic_names]
    labels = [os.path.join(label_dir, id + label_suffix) for id in basic_names]
    return datas, labels


def read_anns(anns, image_dir):
    '''
    read images and labels, return image_list and label_list (file path)
    '''
    with open(anns, 'r') as f:
        res = json.loads(f.read())
    datas = [ os.path.join(image_dir, x['image_name']) for x in res]
    labels = [ os.path.join(image_dir, x['mask_name']) for x in res]
    return datas, labels

def transform_mask(label, threshold=100):
    '''
    input(numpy.narray): shape [H, W]
        if class number > 2, value means the class id
        if class number = 2, 0: background, 1: target
    threshold(int): < threshold background, > threshold target
    '''
    res = np.ones((label.shape[0], label.shape[1]))
    res = label[:,:,0] if len(label.shape) == 3 else label
    if label.max()<threshold:
        return res
    else:
        res[res <= threshold] = 0
        res[res > threshold] = 1
    return res

def trans(image, label, normalize):
    '''
    input: numpy.narray
    '''
    image_out = transforms.ToTensor()(image)
    if normalize:
        image_out = transforms.Normalize(
            normalize["mean"],
            normalize["std"]
            )(image_out)
    lable_out = torch.from_numpy(label)
    return image_out, lable_out

def resize_img(img_pil, scale, type, resize_size):
    """
    input: PIL Image
    return: PIL Image
    """
    w = img_pil.size[0]
    h = img_pil.size[1]
    if not resize_size:
        nw = int(w * scale)
        nh = int(h * scale)
    else:
        nw, nh = resize_size
    if type == "image":
        img_pil = img_pil.resize((nw, nh), Image.ANTIALIAS)
    elif type == "label":
        img_pil = img_pil.resize((nw, nh), Image.NEAREST)
    else:
        img_pil = img_pil.resize((nw, nh), Image.ANTIALIAS)
    return img_pil
    

def crop_img(im, size):
    """
    crop image
    """
    w = im.size[0]
    h = im.size[1]
    return im.crop(map(int, [w*(0.5-0.5/size), h*(0.5-0.5/size), w*(0.5+0.5/size), h*(0.5+0.5/size)]))