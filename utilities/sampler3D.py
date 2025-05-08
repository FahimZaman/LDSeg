"""
@author: Fahim Ahmed Zaman
"""

import numpy as np
import os
from tqdm import tqdm
from utilities.misc import addNoise, mean_variance_normalization, plot_seg_3D, plot_sampling_3D
from utilities.misc import noiseScheduler, plot_noise_parameters

#%%
''' 
This module is used for segmentation using samplers.

Two choices of samplers are available: 
    1. 'DDPM'
    2. 'DDIM'

'''

#%%
def segment(data,
            labelDecoder, 
            imageEncoder, 
            denoiser, 
            model_directory = './savedModels', 
            totalTimesteps = 1000, 
            samplingSteps = 10, 
            sampler = 'DDIM',
            scheduler = 'cosine',
            sigma = 0, 
            batch_size = 4,
            plots = True):                 
    
    # nosie scheduler
    alpha_cumprod, alphas, betas, times = noiseScheduler(T_all = totalTimesteps,
                                                   total_timesteps = samplingSteps)
    alpha_cumprod1 = np.roll(alpha_cumprod, 1)
    
    # sampling parameter plot
    plot_noise_parameters(times, alpha_cumprod, betas, alphas, 
                          schedule = scheduler, sampler=True)
    
    # load model weights
    models = (labelDecoder, imageEncoder, denoiser)
    modelNames = ['labelDecoder3D.hdf5', 'imageEncoder3D.hdf5', 'denoiser3D.hdf5']
    savedmodels = os.listdir(model_directory)
    if all(item in savedmodels for item in modelNames):
        for modl, path in zip(models, modelNames):
            modl.load_weights(os.path.join(model_directory, path))
            print(f'\n{os.path.splitext(path)[0]} weights loaded...')
    else:
        raise Exception('Model weights are unavailable. Please train LDSeg...')
    
    # noise add to images
    if sigma!=0:
        data = addNoise(data, sigma)
    
    # latent spaces        
    z_i = imageEncoder.predict(data, batch_size = batch_size, verbose=0)
    z_lt = mean_variance_normalization(np.random.normal(0, 1, z_i.shape))
    z_lts = []
    z_lts.append(z_lt)
    
    # iteration from T to 1
    for t, beta, acum, acum1 in tqdm(reversed(list(zip(times, betas, alpha_cumprod, alpha_cumprod1)))):
        z_n = denoiser.predict([z_lt, z_i, np.repeat(t, z_lt.shape[0], axis=0)],
                               batch_size=batch_size, verbose=0)
        
        epsilon = np.random.normal(scale=1, size=z_lt.shape)
        if t==times[0]:
            z_lt = (z_lt-np.sqrt(1-acum)*z_n)/np.sqrt(acum) + np.sqrt(beta)*epsilon
        else:
            if sampler=='DDPM':
                s = np.sqrt(((1-acum1)/(1-acum))*(1-acum/acum1))
            else:
                s = 0
            z_lt = np.sqrt(acum1)*((z_lt-np.sqrt(1-acum)*z_n)/np.sqrt(acum)) + np.sqrt(1-acum1-s**2)*z_n + s*epsilon
        z_lt = mean_variance_normalization(z_lt)
        z_lts.append(z_lt)
    
    # plot reverse sampling
    plot_sampling_3D(np.array(z_lts))
    
    # final prediction
    yhat = np.argmax(labelDecoder.predict(z_lt, batch_size = batch_size, verbose=0), axis=-1)
    
    # plots
    if plots==True:
        # number of plots
        if len(data)>=5:
            nplot = 5
        else:
            nplot = len(data)
        plot_seg_3D(data, yhat, nplot)
    return yhat