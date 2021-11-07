#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:40:16 2019

@author: tao
"""

import torch
import torch.nn as nn


def up(x):
    return nn.functional.interpolate(x, scale_factor=2)


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

##############################################
# define our model
class Multi_one_modal_generator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf, num_classes):
        super(Multi_one_modal_generator, self).__init__()

        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc

        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn2 = nn.ReLU(inplace=True)  # nn.ReLU()

        # ~~~ Encoding Paths ~~~~~~ #
        # Encoder (Modality 1)
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim), act_fn,
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim), act_fn,
        )
        self.pool_1 = maxpool()

        self.down_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 2), act_fn,
            nn.Conv2d(in_channels=self.out_dim * 2, out_channels=self.out_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 2), act_fn,
        )
        self.pool_2 = maxpool()

        self.down_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 2, out_channels=self.out_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 4), act_fn,
            nn.Conv2d(in_channels=self.out_dim * 4, out_channels=self.out_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 4), act_fn,
        )
        self.pool_3 = maxpool()

        ## 维度调整
        self.down_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 4, out_channels=self.out_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_dim * 4), act_fn,
            nn.Conv2d(in_channels=self.out_dim * 4, out_channels=self.out_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 4), act_fn,
        )

        ## 分类头
        self.down_fu_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 4, out_channels=self.out_dim * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 8),
            act_fn,
            nn.Conv2d(in_channels=self.out_dim * 8, out_channels=self.out_dim * 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 16),
            act_fn,
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 2),
            # nn.Softmax(),
        )

    def forward(self, inputs):
        down_1 = self.down_1(inputs)
        down_1m = self.pool_1(down_1)

        down_2 = self.down_2(down_1m)
        down_2m = self.pool_2(down_2)

        down_3 = self.down_3(down_2m)
        down_3m = self.pool_3(down_3)

        down_4 = self.down_4(down_3m)

        ## 添加的分类部分
        down_fu_5 = self.down_fu_5(down_4)
        output = self.avg_pool(down_fu_5)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output, down_4





