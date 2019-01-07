#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 10:23:27 2018

@author: peng
"""
from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet
from datagen import ListDataset

from torch.autograd import Variable
import copy

lr=1e-4
resume=False
fix='head'

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

testset = ListDataset(list_file='/mnt/ssd_disk/naka247/peng/DOTA/val_retina_dota_nd.txt', train=False, transform=transform, input_size=1024)
testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=8, collate_fn=testset.collate_fn)


test_loss = 0
for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
    #need to check 10_G272_15Nov2016_0019.JPG, batch_idx=5
    #inputs = Variable(inputs.cuda(), volatile=True)
    #loc_targets = Variable(loc_targets.cuda())
    cls_targets = Variable(cls_targets.cuda())

    pos = cls_targets > 0  # [N,#anchors]
    num_pos = pos.data.long().sum()
    if num_pos==0:
        print(batch_idx)
       # break
