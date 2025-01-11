#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fahim Ahmed Zaman
"""

import numpy as np
import os
import cv2
from tqdm import tqdm
import SimpleITK as sitk
import imageio.v2 as imageio
from utilities.misc import plot_data, standardize
from natsort import natsorted
from skimage.measure import label

#%%
''' 
This module is used for reading and writing images
'''

#%% data read
def readFiles(imgpath, lblpath, size=(512, 512)):
    '''
    Read files given dataset

    Parameters
    ----------
    imgpath : source image path
    lblpath : source label path
    '''
    
    # data lists
    img_array, lbl_array = [], []
    
    # read data
    for n in tqdm(range(len(imgpath))):
        img_array.append(standardize(cv2.resize(imageio.imread(imgpath[n]), size, 0, 0, interpolation = cv2.INTER_NEAREST)))
        lbl = cv2.resize(imageio.imread(lblpath[n]), size, 0, 0, interpolation = cv2.INTER_NEAREST)
        lbl[lbl>0] = 1
        lbl_array.append(lbl)
    img_array = np.array(img_array).astype(np.float32)
    lbl_array = np.array(lbl_array).astype(np.uint8) 
    return img_array, lbl_array

def dataRead(dataset='GlaS'):
    '''Read data given data path'''
    
    # dataset path
    filepath = os.path.join('./Data', dataset)
    images = os.path.join(filepath,'Image')
    labels = os.path.join(filepath,'Label')
    imgpath, lblpath = natsorted([os.path.join(images,i) for i in os.listdir(images)]), natsorted([os.path.join(labels,i) for i in os.listdir(labels)])
    img_array, lbl_array = readFiles(imgpath, lblpath)
    print('\nData shape: ', img_array.shape[1:])
    
    # plot
    plot_data(img_array, lbl_array)
    return img_array, lbl_array

#%% data write
def writeImage(segs, imgpath, segpath):
    '''Write segmentations with interpolation'''
    for n in tqdm(range(len(segs))):
        H, W, _ = imageio.imread(imgpath[n]).shape
        lbl = cv2.resize(segs[n], (W, H), 0, 0, interpolation = cv2.INTER_NEAREST)
        lbl = np.where(lbl>0,1,0)
        lbl = label(lbl, connectivity=2).astype(np.uint8)
        imageio.imwrite(segpath[n], lbl)
        print('\nWriting to '+segpath[n])
            
def segWrite(segmentation):
    '''Write segmentations'''
    
    # dataset path
    filepath=os.path.join('./Data/GlaS')
    imgdir = os.path.join(filepath,'Image')
    segdir = os.path.join(filepath,'Segmentation')
    if not os.path.exists(segdir):
        os.makedirs(segdir)
    
    # filenames
    tag = '_segmentation.'
    imgfilenames = os.listdir(imgdir)
    imgpath = [os.path.join(imgdir, i) for i in imgfilenames]
    segfilenames = natsorted([i.split(os.extsep,1)[0]+tag+i.split(os.extsep,1)[1] for i in imgfilenames])
    segpath = [os.path.join(segdir, i) for i in segfilenames]
    
    # write segmentations
    writeImage(segmentation, imgpath, segpath)