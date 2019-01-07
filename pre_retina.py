#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:32:22 2017

@author: peng
"""

########this script is to prepare data for faster-rcnn keras version
# keras version is not perfect but it is easiest to understand and get result
# the backend is using tensorflow!

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets, models
import random
import PIL
from PIL import Image, ImageOps
import math
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import cv2
import skimage.feature

import glob

'''
root_dir='/home/peng/Desktop/final_round_experiments/easy/'
image_folder='train11_512'
target_folder='train11_512_dot'
output='train_retina.txt'
'''
def get_input_files(root_dir,image_folder,target_folder,output):
    img_path = os.path.join(root_dir, image_folder)
    img_list=sorted(glob.glob(img_path+'/*.JPG')) 
    target_path = os.path.join(root_dir, target_folder)
    target_list=sorted(glob.glob(target_path+'/*.JPG')) 
    
    
    all_cords=[]
    for idx in range(len(img_list)):      
        image = io.imread(img_list[idx])
        target = io.imread(target_list[idx])     
        
        #break
        
        image_3 = cv2.absdiff(target,image)
        mask_1 = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
        mask_1[mask_1 < 20] = 0
        mask_1[mask_1 > 0] = 255
        
        mask_2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask_2[mask_2 < 20] = 0
        mask_2[mask_2 > 0] = 255
        
        image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
        image_5 = cv2.bitwise_or(image_4, image_4, mask=mask_2) 
        
        # convert to grayscale to be accepted by skimage.feature.blob_log
        image_6 = cv2.cvtColor(image_5, cv2.COLOR_RGB2GRAY)
        # detect blobs
        blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=10, num_sigma=1, threshold=0.05)
        
        if len(blobs)!=0:
            m=10
            nneg = lambda x: max(0, x)
            cord=[img_list[idx]]
            for blob in blobs:
                # get the coordinates for each blob
                y, x, s = blob
                # assign bounding box to image
                iy=int(y)
                ix=int(x)
                
                x1=nneg(ix-m)
                y1=nneg(iy-m)
                x2=nneg(ix+m)
                y2=nneg(iy+m)
                cord.extend([x1,y1,x2,y2])
                cord.extend([1])
            all_cords.append(cord)
    outF = open(os.path.join('/mnt/ssd_disk/naka247/peng/pytorch-retinanet-master',output), "w")
    for line in all_cords:
      # write line to output file
      
      outF.write(str(line)[1:len(str(line))-1])
      outF.write("\n")
    outF.close()           
                

        
        
#####segmentation blob detection        
get_input_files('/mnt/ssd_disk/naka247/peng/bird/hard/','train11_512','train11_512_dot','train_retina_hard.txt')               
get_input_files('/mnt/ssd_disk/naka247/peng/bird/hard/','val11_512','val11_512_dot','val_retina_hard.txt')              
get_input_files('/mnt/ssd_disk/naka247/peng/bird/hard/','test11_512','test11_512_dot','test_retina_hard.txt')                      
                 
