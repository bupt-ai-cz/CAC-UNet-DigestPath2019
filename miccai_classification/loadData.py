#
# author: Sachin Mehta
# Project Description: This repository contains source code for semantically segmenting WSIs; however, it could be easily
#                   adapted for other domains such as natural image segmentation
# File Description: This file is used to check and pickle the data
# ==============================================================================

import numpy as np
import os.path
from PIL import Image
import cv2
import pickle
import json
import pdb


class LoadData:
    def __init__(self, data_dir, classes, cached_data_file, normVal=1.10):
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.trainImList = list()
        self.valImList = list()
        self.trainAnnotList = list()
        self.valAnnotList = list()
        self.trainClass = list()
        self.valClass = list()

        self.cached_data_file = cached_data_file

    def compute_class_weights(self, histogram):
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readFile(self, fileName, trainStg=False):
        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            textFile = json.load(textFile)
            for line in textFile:
                # pdb.set_trace()
                # if line['label']==1:
                #     img_file = '../miccai/tissue-train-pos/'+line['name'].strip()
                # else:
                #     img_file = '../miccai/crop_neg/' + line['name'].strip()

                # 根据目前的train_classify.json和val_classify_5.json修改了路径构造规则

                # TODO: 如果重新生成了json文件，注意这里的路径构造代码也要修改
                if line['name'].__contains__('crop_neg/'):
                    img_file = os.path.join('../miccai', line['name'].strip())
                else:
                    img_file = os.path.join('../miccai/tissue-train-pos', line['name'].strip())

                if trainStg:
                    self.trainImList.append(img_file)
                    self.trainClass.append(line['label'])
                else:
                    self.valImList.append(img_file)
                    self.valClass.append(line['label'])
        return 0

    def processData(self):
        # TODO: 如果重新生成了json文件，记得修改路径

        print('Processing training data')
        # 未找到这个10fold目录
        # return_val = self.readFile('10fold/train_classify_5.json', True)
        return_val = self.readFile('train_classify.json', True)

        print('Processing validation data')
        # return_val1 = self.readFile('10fold/val_classify_5.json')
        return_val1 = self.readFile('val_classify.json')

        print('Pickling data')
        if return_val == 0 and return_val1 == 0:
            # if return_val1 ==0:
            data_dict = dict()
            data_dict['trainIm'] = self.trainImList
            data_dict['trainClass'] = self.trainClass
            data_dict['valIm'] = self.valImList
            data_dict['valClass'] = self.valClass
            pickle.dump(data_dict, open(self.cached_data_file, "wb"))
            print(len(data_dict['trainIm']), len(data_dict['valIm']))
            return data_dict
        return None
