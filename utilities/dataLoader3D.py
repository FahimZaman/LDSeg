#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fahim Ahmed Zaman
"""

import numpy as np
import os
from scipy.ndimage import zoom
from tqdm import tqdm
import SimpleITK as sitk
from utilities.misc import plot_data_3D, standardize
from natsort import natsorted

#%%
''' 
This module is used for reading and writing 3D images
'''

#%% data read
def interp(volume, D, H, W):
    '''
    This module is used for 3D interpolation

    Parameters
    ----------
    volume : 3D volume
    D : Depth
    H : Height
    W : Width

    Returns
    -------
    resampled_volume : interpolated 3D volume

    '''
    d, h, w = volume.shape
    resampling_factor = (D/d, H/h, W/w)
    resampled_volume = zoom(volume, resampling_factor, order=1, mode='nearest')
    return resampled_volume

def readFiles(imgpath, lblpath, size=(128, 128, 256)):
    '''
    Read files given dataset

    Parameters
    ----------
    imgpath : source image path
    lblpath : source label path
    size : (Depth, Height, Width)
    '''
    
    # data lists
    img_array, lbl_array = [], []
    
    # read data
    for n in tqdm(range(len(imgpath))):
        img = sitk.GetArrayFromImage(sitk.ReadImage(imgpath[n])).astype(np.float32)
        img = standardize(img)
        seg = sitk.GetArrayFromImage(sitk.ReadImage(lblpath[n])).astype(np.uint8)
        seg1, seg2 = np.where(seg==1, 1, 0), np.where(seg==2, 1, 0)
        img, seg1, seg2 = (interp(img, size[0], size[1], size[2]), 
                           interp(seg1, size[0], size[1], size[2]), 
                           interp(seg2, size[0], size[1], size[2]))
        seg2[seg2>0]=1
        seg1[seg2>0]=2
        img_array.append(img)
        lbl_array.append(seg1)
    img_array = np.expand_dims(np.array(img_array), axis=-1).astype(np.float32)
    lbl_array = np.array(lbl_array).astype(np.uint8) 
    return img_array, lbl_array

def dataRead(dataset='Knee'):
    '''Read data given data path'''
    
    # dataset path
    filepath = os.path.join('./Data', dataset)
    images = os.path.join(filepath,'Image')
    labels = os.path.join(filepath,'Label')
    imgpath, lblpath = natsorted([os.path.join(images,i) for i in os.listdir(images)]), natsorted([os.path.join(labels,i) for i in os.listdir(labels)])
    img_array, lbl_array = readFiles(imgpath, lblpath)
    print('\nData shape: ', img_array.shape[1:])
    
    # plot
    plot_data_3D(img_array, lbl_array)
    return img_array, lbl_array

#%% data write
def writeImage(segs, imgpath, segpath):
    '''Write segmentations with interpolation'''
    for n in tqdm(range(len(segs))):
        img = sitk.GetArrayFromImage(sitk.ReadImage(imgpath[n])).astype(np.float32)
        D, H, W = img.shape
        seg1, seg2 = np.where(segs[n]==1, 1, 0), np.where(segs[n]==2, 1, 0)
        seg1, seg2 = interp(seg1, D, H, W), interp(seg2, D, H, W)
        seg2[seg2>0]=1
        seg1[seg2>0]=2
        sitk.WriteImage(sitk.GetImageFromArray(seg1.astype(np.uint8)), segpath[n])
        print('\nWriting to '+segpath[n])
            
def segWrite(segmentation):
    '''Write segmentations'''
    
    # dataset path
    filepath=os.path.join('./Data/Knee')
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