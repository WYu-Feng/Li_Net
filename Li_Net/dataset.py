#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tus April 18 17:18:50 2019

@author: tao
"""
#coding:utf8
import os
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from funcs.utils import *
import torch
import scipy.io as scio
import glob
import cv2
from PIL import Image
from imgaug import augmenters as iaa  # 引入数据增强的包
import random

def loadSubjectData(path1, path2, if_train = True):
    
    img_t1 = cv2.imread(path1)
    img_t2 = cv2.imread(path2)

    if if_train:
        img_t1, img_t2 = augment(img_t1, img_t2)
    else:
        img_t1 = Image.fromarray(np.uint8(img_t1))
        img_t2 = Image.fromarray(np.uint8(img_t2))

    # img_t1 = Image.fromarray(np.uint8(img_t1))
    # img_t2 = Image.fromarray(np.uint8(img_t2))
    img_t1 = np.array(img_t1.resize(size=(512, 512)))
    img_t2 = np.array(img_t2.resize(size=(512, 512)))

    return img_t1,img_t2

def augment(img1, img2):

    def crop(image, x1, x2, x3, x4):
        cropped = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        cropped[x1:image.shape[0] - x2, x3:image.shape[0] - x4, :] = image[x1:image.shape[0] - x2, x3:image.shape[0] - x4, :]
        return cropped

    def flip(image, if_lr, if_ud):
        if if_lr:
            image = np.fliplr(image).copy()
        if if_ud:
            image = np.flipud(image).copy()
        return image

    # def rotate(image, angle):
    #     image2 = rotate(image, angle).astype(np.uint8)
    #     return image2
    img1_x , img1_y = img1.shape[0], img1.shape[1]
    x1, x2, x3, x4 = random.randint(int(0.05 * img1_x), int(0.1 * img1_x)), \
                     random.randint(int(0.05 * img1_x), int(0.1 * img1_x)), \
                     random.randint(int(0.05 * img1_y), int(0.1 * img1_y)), \
                     random.randint(int(0.05 * img1_y), int(0.1 * img1_y))
    if_lr, if_ud = random.randint(1, 3)==2, random.randint(1, 3)==2
    # angle = random.randint(0, 15)

    img1 = crop(img1, x1, x2, x3, x4)
    img2 = crop(img2, x1, x2, x3, x4)

    # img_aug1 = flip(img1, if_lr, if_ud)
    # img_aug2 = flip(img2, if_lr, if_ud)

    # img_aug1 = rotate(img1, angle)
    # img_aug2 = rotate(img2, angle)
    img_aug1 = Image.fromarray(np.uint8(img1))
    img_aug2 = Image.fromarray(np.uint8(img2))
    return img_aug1, img_aug2

class MultiModalityData_load(data.Dataset):
    
    def __init__(self,opt,transforms=None,if_train=True):
        
        self.opt   = opt
        self.if_train = if_train
        self.label = list()
        self.img_t1_path = list()
        self.img_t2_path = list()

        if self.if_train:
            img_t11_path =glob.glob(os.path.join('dataset', opt.data_path, 'class1_1', '*.jpg'))
            img_t12_path = glob.glob(os.path.join('dataset', opt.data_path, 'class1_2', '*.jpg'))
            train_data1_len = int(len(img_t11_path) * self.opt.train_rate)

            self.img_t1_path = self.img_t1_path + img_t11_path[:train_data1_len]
            self.img_t2_path = self.img_t2_path + img_t12_path[:train_data1_len]
            self.label = self.label + [0 for _ in range(train_data1_len)]

            img_t21_path =glob.glob(os.path.join('dataset', opt.data_path, 'class2_1', '*.jpg'))
            img_t22_path = glob.glob(os.path.join('dataset', opt.data_path, 'class2_2', '*.jpg'))
            train_data2_len = int(len(img_t21_path) * self.opt.train_rate)

            self.img_t1_path = self.img_t1_path + img_t21_path[:train_data2_len]
            self.img_t2_path = self.img_t2_path + img_t22_path[:train_data2_len]
            self.label = self.label + [1 for _ in range(train_data1_len)]

        else:
            img_t11_path =glob.glob(os.path.join('dataset', opt.data_path, 'class1_1', '*.jpg'))
            img_t12_path = glob.glob(os.path.join('dataset', opt.data_path, 'class1_2', '*.jpg'))
            train_data1_len = int(len(img_t11_path) * self.opt.train_rate)

            self.img_t1_path = self.img_t1_path + img_t11_path[train_data1_len:]
            self.img_t2_path = self.img_t2_path + img_t12_path[train_data1_len:]
            self.label = self.label + [0 for _ in range(len(img_t11_path) - train_data1_len)]

            img_t21_path =glob.glob(os.path.join('dataset', opt.data_path, 'class2_1', '*.jpg'))
            img_t22_path = glob.glob(os.path.join('dataset', opt.data_path, 'class2_2', '*.jpg'))
            train_data2_len = int(len(img_t21_path) * self.opt.train_rate)

            self.img_t1_path = self.img_t1_path + img_t21_path[train_data2_len:]
            self.img_t2_path = self.img_t2_path + img_t22_path[train_data2_len:]
            self.label = self.label + [1 for _ in range(len(img_t21_path) - train_data2_len)]

    def __getitem__(self,index):
        
        # path
        cur_img_t1_path = self.img_t1_path[index]
        cur_img_t2_path = self.img_t2_path[index]

        # get images
        img_t1, img_t2 = loadSubjectData(cur_img_t1_path, cur_img_t2_path, self.if_train)

        tensor_img1 = F.to_tensor(img_t1) * 2 - 1.
        tensor_img2 = F.to_tensor(img_t2) * 2 - 1.
        label = np.array(self.label[index])

        return tensor_img1, tensor_img2, label
    
    
    
    def __len__(self):
        return len(self.label)
    
     
