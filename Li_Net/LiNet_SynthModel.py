import os
import torch
import torch.nn as nn
import numpy as np

import time
import datetime

from torch.utils.data import DataLoader
from dataset import MultiModalityData_load
from funcs.utils import *
import torch.nn as nn
import scipy.io as scio
from torch.autograd import Variable
import torch.autograd as autograd
import model.syn_model as models
import model.syn_one_model as one_model


cuda = True if torch.cuda.is_available() else False
FloatTensor   = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor    = torch.cuda.LongTensor if cuda else torch.LongTensor


class LatentSynthModel():
    
    ########################################################################### 
    
    def __init__(self,opt):
        self.opt         = opt  
        self.generator   = models.Multi_modal_generator(3, 3, 32, 2)
        self.generator_one_model = one_model.Multi_one_modal_generator(3, 3, 32, 2)
        self.model_id = None

        if opt.use_gpu: 
            self.generator    = self.generator.cuda()
            self.generator_one_model = self.generator_one_model.cuda()

    ########################################################################### 

    ###### 多模态训练
    def train(self):
        self.generator.apply(weights_init_normal)
        optimizer_G     = torch.optim.Adam(self.generator.parameters(),lr=self.opt.multimodel_lr, betas=(self.opt.b1, self.opt.b2))

        # Learning rate update schedulers
        lr_scheduler_G  = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.opt.epochs, 0, self.opt.decay_epoch).step)
    
            
        # Lossesgenerator
        criterion_identity = nn.L1Loss().cuda()
        criterion_CEL = nn.BCELoss().cuda()

        # Load data        
        train_data   = MultiModalityData_load(self.opt, if_train=True)
        train_loader = DataLoader(train_data,batch_size=self.opt.batch_size,shuffle=True)

        # ---------------------------- *training * ---------------------------------
        best_acc = 0
        for epoch in range(self.opt.epochs):
            for ii, inputs in enumerate(train_loader):
                if self.opt.use_gpu:
                    x1 = inputs[0].cuda()
                    x2 = inputs[1].cuda()
                    y = inputs[2].type(torch.FloatTensor)
                    onehot_y = torch.eye(2)[y.long(), :].cuda()

                x_fu = torch.cat([x1,x2],dim=1)
                optimizer_G.zero_grad()
                label, x1_re, x2_re, _ = self.generator(x_fu)
                label = nn.functional.softmax(label, dim = 1)
                loss_label = criterion_CEL(label, onehot_y)

                # Identity loss
                loss_re1 = criterion_identity(x1_re, x1)
                loss_re2 = criterion_identity(x2_re, x2)

                loss_G = 10 * loss_label + loss_re1 + loss_re2

                loss_G.backward()
                optimizer_G.step()
            # Update learning rates
            lr_scheduler_G.step()

            acc = self.eval(epoch)
            ### 保存最高精度的模型
            if acc > best_acc:
                best_acc = acc
                torch.save(self.generator.state_dict(), "best_multimodel.pth")

    def eval(self, epoch):
        train_data   = MultiModalityData_load(self.opt, if_train=False)
        train_loader = DataLoader(train_data,batch_size=1,shuffle=False)

        self.generator.eval()
        with torch.no_grad():
            correct = 0
            for ii, inputs in enumerate(train_loader):
                if self.opt.use_gpu:
                    x1 = inputs[0].cuda()
                    x2 = inputs[1].cuda()
                    y = inputs[2].type(torch.FloatTensor).cuda()
                x_fu = torch.cat([x1,x2],dim=1)
                label, _, _, _ = self.generator(x_fu)

                for label_item, y_item in zip(label, y):
                    if label_item[0] > label_item[1] and y_item == 0:
                        correct += 1
                    elif label_item[0] < label_item[1] and y_item == 1:
                        correct += 1

            acc = correct/len(train_loader)
            print('multimodel epoch:{:}_acc:{:}'.format(epoch, acc))
        self.generator.train()
        return acc

    ###### 单模态蒸馏
    def train_one_model(self, KD = True, model_id = 1):
        ## 加载最优多模态模型，用于数据蒸馏（固定参数）
        self.generator.load_state_dict(torch.load("best_multimodel.pth"))
        self.generator.eval()
        self.generator_one_model.apply(weights_init_normal)

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator_one_model.parameters(), lr=self.opt.single_lr, betas=(self.opt.b1, self.opt.b2))
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.opt.epochs, 0,
                                                                                           self.opt.decay_epoch).step)

        criterion_CEL = nn.BCELoss().cuda()

        # Load data
        train_data = MultiModalityData_load(self.opt, if_train=True)
        train_loader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True)

        # ---------------------------- *training * ---------------------------------
        for epoch in range(self.opt.epochs):
            for ii, inputs in enumerate(train_loader):
                if self.opt.use_gpu:
                    if self.model_id == 1:
                        x_fu_one_model = inputs[0].cuda()
                    else:
                        x_fu_one_model = inputs[1].cuda()
                    x1 = inputs[0].cuda()
                    x2 = inputs[1].cuda()
                    x_fu = torch.cat([x1, x2], dim=1)
                    y = inputs[2].type(torch.FloatTensor)
                    onehot_y = torch.eye(2)[y.long(), :].cuda()

                optimizer_G.zero_grad()
                label, fe_one_model = self.generator_one_model(x_fu_one_model)
                
                if KD == True:
                    with torch.no_grad():
                        teacher_outputs, _, _, fe = self.generator(x_fu)
                    loss_kd = criterion_CEL(nn.functional.softmax(label/0.8, dim = 1), nn.functional.softmax(teacher_outputs/0.8, dim = 1))
                    dist = preceptual_loss(fe, fe_one_model)
                    loss_G = loss_kd + dist
                else:
                    label = nn.functional.softmax(label, dim = 1)
                    loss_G = criterion_CEL(label, onehot_y)                    

                loss_G.backward()
                optimizer_G.step()

            self.eval_one_model(epoch)
            lr_scheduler_G.step()

    def eval_one_model(self, epoch):
        test_data   = MultiModalityData_load(self.opt, if_train=False)
        test_loader = DataLoader(test_data,batch_size=10,shuffle=True)
        self.generator_one_model.eval()
        with torch.no_grad():
            correct = 0
            n = 0
            for ii, inputs in enumerate(test_loader):
                if self.opt.use_gpu:
                    if self.model_id == 1:
                        x_fu = inputs[0].cuda()
                    else:
                        x_fu = inputs[1].cuda()
                    y = inputs[2].type(torch.FloatTensor).cuda()
                label, _ = self.generator_one_model(x_fu)

                for label_item, y_item in zip(label, y):
                    n += 1
                    if label_item[0] > label_item[1] and y_item == 0:
                        correct += 1
                    elif label_item[0] < label_item[1] and y_item == 1:
                        correct += 1
            acc = correct/n
            print('single model epoch:{:}_acc:{:}'.format(epoch, acc))
        self.generator_one_model.train()

    