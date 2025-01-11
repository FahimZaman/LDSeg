"""
@author: Fahim Ahmed Zaman
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import subprocess as sp
import tensorflow as tf

#%%
''' 
This module is used for miscellaneous requirements

'''

#%% miscellaneous
def setGPU(multiple=True):
    # check if GPU available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if multiple==False:
            command = "nvidia-smi --query-gpu=memory.free --format=csv"
            memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
            # available memory in GPU
            memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
            # get GPU with maximum memory
            gpu_number = np.argmax(memory_free_values)
            # set gpu
            tf.config.set_visible_devices(gpus[gpu_number], 'GPU')
            print('\nGPU:\n', tf.config.get_visible_devices('GPU'))
        else:
            print('\nGPU:\n', tf.config.get_visible_devices('GPU'))
    else:
        print('\nGPU is not vailable, running on CPU...\n')

def standardize(img):
    '''Standardize image [0..1]'''
    img = img - np.min(img)
    img = img / np.max(img)
    return img  

def addNoise(img, sigma):
    '''Add noise to images given sigma'''
    noise = np.random.normal(scale=sigma, size=img.shape)
    nvol = standardize(img + noise)
    return nvol

def mean_variance_normalization(x, scale=1.0):
    '''Normalize data''' 
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean) / (std + 1e-5)  # Normalize to zero mean and unit variance
    x = x * scale  # Scale to desired range
    return x

def cosineFunc(t, T, s=0.008):
    '''Returns cosine curve'''
    return np.cos(((t/T+s)/(1+s))*(np.pi/2))**2

def noiseScheduler(T_all=1000, total_timesteps=15):
    '''Noise scheduler parameters'''
    times = np.linspace(0, T_all-1, total_timesteps)
    alpha_cumprod = cosineFunc(times, T_all)/cosineFunc(np.zeros_like(times), T_all)
    alpha_cumprod1 = np.append(1.0, alpha_cumprod[:-1])
    alphas = alpha_cumprod/alpha_cumprod1
    betas = 1 - (alphas)
    return alpha_cumprod, alphas, betas, times

def plot_data(img_array, lbl_array):
    '''Plots sample image from dataset'''
    idx = random.randint(0,len(img_array)-1)
    plt.imshow(img_array[idx])
    show_mask(np.where(lbl_array[idx]==1,1,0), alpha=0.35, random_color=4)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_sampling(hm_intermediates, idx=0):
    '''Plots reverse diffusion steps given intermediate denoised images'''
    sampling_steps = len(hm_intermediates)
    if sampling_steps>=10:
        sampling_steps = 10
    steps = (np.linspace(0, len(hm_intermediates)-1, sampling_steps).astype(np.uint64))
    plt.figure(figsize=(3*sampling_steps,5))
    for i in range(len(steps)):
        img = hm_intermediates[steps[i]][idx]
        plt.subplot(1,sampling_steps,i+1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle('Reverse Diffusion', fontsize=5*sampling_steps)
    plt.tight_layout()
    plt.show()
    
def plot_noise_parameters(times, alphas_cumprod, betas, alphas, schedule='cosine', sampler=False):
    '''Plots noise parameters: alpha_cumulative_products, alphas and betas'''
    fontsize = 15
    plt.figure(figsize=(10,3))
    # forward diffusion parameters
    if sampler==False:
        plt.subplot(1,3,1)
        plt.plot(times, alphas_cumprod)
        plt.title(r'$\overline{\alpha}$', fontsize=fontsize)
        plt.xlabel('timesteps')
        plt.subplot(1,3,2)
        plt.plot(times, betas)
        plt.title(r'$\beta$', fontsize=fontsize)
        plt.xlabel('timesteps')
        plt.subplot(1,3,3)
        plt.plot(times, alphas)
        plt.title(r'$\alpha$', fontsize=fontsize)
        plt.xlabel('timesteps')
        plt.suptitle('Noise Parameters [schedule: '+schedule+']', fontsize=fontsize+5)
        plt.tight_layout()
    # reverse diffusion parameters
    else:
        plt.figure(figsize=(10,3))
        plt.subplot(1,3,1)
        plt.scatter(times, alphas_cumprod)
        plt.title(r'$\overline{\alpha}$', fontsize=fontsize)
        plt.xlabel('timesteps')
        plt.xlim(np.max(times),0)
        plt.subplot(1,3,2)
        plt.scatter(times, betas)
        plt.title(r'$\beta$', fontsize=fontsize)
        plt.xlabel('timesteps')
        plt.xlim(np.max(times),0)
        plt.subplot(1,3,3)
        plt.scatter(times, alphas)
        plt.title(r'$\alpha$', fontsize=fontsize)
        plt.xlabel('timesteps')
        plt.xlim(np.max(times),0)
        plt.suptitle('Sampling Parameters [schedule: '+schedule+']', fontsize=fontsize+5)
    plt.tight_layout()
    plt.show()
    
def show_mask(mask, alpha=1, random_color=1, ax=0):
    if random_color==1:
        color = np.array([1, 0, 0, alpha])
    elif random_color==2:
        color = np.array([0, 1, 0, alpha])
    elif random_color==3:
        color = np.array([0, 0, 1, alpha])
    elif random_color==4:
        color = np.array([1, 1, 0, alpha])
    else:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if ax!=0:
        ax.imshow(mask_image, alpha=1)
    else:
        plt.imshow(mask_image, alpha=1)
        
def plot_seg(img, prediction, nplot):
    '''Plot segmentation results'''
    idx = random.sample(range(0, len(img)), nplot)
    plt.figure(figsize=(5*nplot,5))
    for i in range(len(idx)):
        plt.subplot(1,nplot,i+1)
        plt.imshow(img[idx[i]])
        show_mask(np.where(prediction[idx[i]]==1,1,0), alpha=0.35, random_color=4)
        plt.title(f'Img-{idx[i]:d}', fontsize=30)
        plt.axis('off')
    plt.tight_layout()
    plt.show()