from PIL import Image,ImageOps
import random
import os.path as osp
import os
import numpy as np
import math
import cv2
from tqdm import tqdm
import glob
import multiprocessing

Image.MAX_IMAGE_PIXELS=None


    
def is_blank(source):
    source = np.array(source).astype('uint8')
    channels = cv2.split(source)
    return all([c.std() < 30 for c in channels])

def do_img_split(suffix, input_path, out_dir, IM_WIDTH, IM_HEIGHT, size=3, is_filter=True, fill="white"):
    origin_im_name = osp.split(input_path)[-1]
    file_basename = osp.splitext(origin_im_name)[0].replace('_mask','')
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    img = Image.open(input_path)
    w,h = img.size
    cols = int(math.ceil(w / IM_WIDTH))
    rows = int(math.ceil(h / IM_HEIGHT))

    # padding
    _right = cols * IM_WIDTH - w
    _bottom = rows * IM_HEIGHT - h
    img = ImageOps.expand(img,(0,0,_right,_bottom),fill=fill)
    img = ImageOps.expand(img,(int(IM_WIDTH*size),int(IM_HEIGHT*size),int(IM_WIDTH*size),int(IM_HEIGHT*size)),fill=fill)
    w = cols * IM_WIDTH
    h = rows * IM_HEIGHT

    # split
    path_list = []
    for i in range(rows):
        for j in range(cols):
            name = file_basename + '_%03dx%03d%s' % (i + 1, j + 1, suffix)
            out_path = osp.join(out_dir, name)
            patch = img.crop((j*IM_WIDTH, i*IM_HEIGHT, int((j+size)*IM_WIDTH), int((i+size)*IM_HEIGHT))).convert('RGB')
            if is_filter:
                if not is_blank(patch):
                    patch.save(out_path)
                    path_list.append((name, i, j))
            else:
                patch.save(out_path)
                path_list.append((name, i, j))
    print(input_path)

def main():
    print('start split!')
    image_list = glob.glob('../data/tissue-train-pos-v1/*[0-9].jpg')
    label_list = glob.glob('../data/tissue-train-pos-v1/*mask.jpg')

    pool = multiprocessing.Pool(processes=6)
    for i in label_list:
        pool.apply_async(do_img_split, args=(
            '_mask.jpg', 
             i,
             '../data/pos-patches/', 
             512,
             512,
             3,
             False,
             "black"
             ))
    for i in image_list:
        pool.apply_async(do_img_split, args=(
            '_img.jpg', 
             i,
             '../data/pos-patches/', 
             512,
             512,
             3,
             True,
             "white"
             ))
    pool.close()
    pool.join()
    
if __name__ == '__main__':
    main()
