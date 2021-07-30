import os
import glob
import json
import random
import cv2
import numpy as np
from tqdm import tqdm

def has_target(i):
    img = i['mask_name']
    img_path = "mydata/tissue-train-pos/" + img
    im = cv2.imread(img_path, 0)
    im = im[512:1024,512:1024]
    res = np.bincount(im.flatten())
    if sum(res[0:150])/(512*512) < 0.95:
        i['has_target'] = True
    else:
        i['has_target'] = False
    return i

all_images = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob('../data/tissue-train-pos-v1/*[0-9].jpg')]
train_list = []
val_list = []
for x in all_images:
    if random.random() > 0.1:
        train_list.append(x)
    else:
        val_list.append(x)

with open('mydata/512x3_anns.json', 'r') as f:
    res = json.loads(f.read())

train = []
val = []
for x in res:
    id_x = os.path.splitext(os.path.basename(x['image_name']))[0][:-12]
    print(id_x)
    if id_x in train_list:
        train.append(x)
    else:
        val.append(x)

with open('mydata/train.json', 'w') as f:
    f.write(json.dumps(train))
with open('mydata/val.json', 'w') as f:
    f.write(json.dumps(val))
print('train: ', len(train))
print('val: ', len(val))
print('finished!')