#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 20:50:23 2018

@author: peng
"""

from __future__ import print_function, division
import os
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import random
import PIL
from PIL import Image, ImageOps
import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import cv2
import skimage.feature

import glob

# =============================================================================
# read fold information
# =============================================================================
for fold in range(1,11):
    fold_type='fold%02d' % fold
    root='/storage/vision247/workspace/peng/VEDAI/Annotations512/'
    trainval_file=os.path.join(root,fold_type+'.txt')
    test_file=os.path.join(root,fold_type+'test.txt')
    
    with open(trainval_file) as f:
        trainval_lines = f.readlines()
    
    with open(test_file) as f:
        test_lines = f.readlines()    
    
    
    # =============================================================================
    # change bb information
    # =============================================================================
    annotations=pd.read_csv(os.path.join(root,'annotation512.txt'),sep=" ", header=None)
    cols=['image_id','x','y','orient','x1','x2','x3','x4','y1','y2','y3','y4','class','Flag1','Flag2']
    annotations.columns = cols
    
    
    #replace class id into 1-9
    # =============================================================================
    # 31:'plane',9
    # 23:'boat',8
    # 5:'camping car',4
    # 1:'car',1
    # 11:'pick-up',7
    # 4:'tractor',3
    # 2:'truck',2
    # 9:'van',5
    # 10:'other',6
    # =============================================================================
    annotations['class']=annotations['class'].replace(1,1)
    annotations['class']=annotations['class'].replace(2,2)
    annotations['class']=annotations['class'].replace(4,3)
    annotations['class']=annotations['class'].replace(5,4)
    annotations['class']=annotations['class'].replace(9,5)
    annotations['class']=annotations['class'].replace(10,6)
    annotations['class']=annotations['class'].replace(11,7)
    annotations['class']=annotations['class'].replace(23,8)
    annotations['class']=annotations['class'].replace(31,9)
    
    trainval_id=[int(name) for name in trainval_lines]
    test_id=[int(name) for name in test_lines]
    
    
    trainval_table=annotations.loc[annotations['image_id'].isin(trainval_id),:]
    test_table=annotations.loc[annotations['image_id'].isin(test_id),:]    
    
    arr=test_table.iloc[0,4:12]
    def bb_convert(arr):
        x_list=np.array(arr)[0:4]
        y_list=np.array(arr)[4:8]
        h=max(y_list)-min(y_list)
        w=max(x_list)-min(x_list)
        x_top=min(x_list)
        y_top=min(y_list)
        return x_top,y_top,x_top+w,y_top+h
        
    test_table['bb']=test_table.iloc[:,4:12].apply(bb_convert,axis=1)    
    trainval_table['bb']=trainval_table.iloc[:,4:12].apply(bb_convert,axis=1)    
    
    # =============================================================================
    # prepare the input data for each model
    # =============================================================================
    
    
    
    
    
    
    ##retina
    image_root='/storage/vision247/workspace/peng/VEDAI/Vehicules512/'
    #output_name='trainval_retina0.txt'
    def pre_retina(data_type,table_type,output_name):
        all_trainval=[]
        for item in data_type:
            image_name=os.path.join(image_root,item.split('\n')[0]+'_co.png')
            image_table=table_type.loc[table_type['image_id']==int(item),:]
            bb_info=image_table['bb'].tolist()
            class_info=image_table['class'].tolist()
            output=[image_name]
            for i in range(len(bb_info)):
                cord=bb_info[i]
                x1=cord[0]
                x2=cord[2]
                y1=cord[1]
                y2=cord[3]
                car_class=class_info[i]
                output.extend([x1,y1,x2,y2])
                output.extend([car_class])
            all_trainval.append(output)
        outF = open(os.path.join('/storage/vision247/workspace/peng/VEDAI/',output_name), "w")
        for line in all_trainval:
          # write line to output file
          
          outF.write(str(line)[1:len(str(line))-1])
          outF.write("\n")
        outF.close() 
    
    pre_retina(trainval_lines,trainval_table,'trainval_retina%02d.txt' % fold)
    pre_retina(test_lines,test_table,'test_retina%02d.txt' % fold)
