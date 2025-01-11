"""
@author: Fahim Ahmed Zaman
"""

import numpy as np
import tensorflow as tf
from utilities.misc import plot_noise_parameters, cosineFunc

#%%
''' 
This module is used for forward diffusion using Gaussian parameterization

Three choices of noise schedules are available: 
    1. 'cosine': beta_0 and beta_T is irrelevant and obtained from alpha_cumulative_product)
    2. 'linear': linear function of beta given beta_0 and beta_T
    3. 'quadratic': nonlinear second order function of beta given beta_0 and beta_T

'''

#%% Gaussian Block For Forward Diffusion
class GaussianDiffusion:
    '''Gaussian block'''
    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        schedule='cosine'):
        
        # Define initial parameters
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        
        # Define the linear variance schedule
        if schedule=='linear':
            self.betas = betas = np.linspace(
                beta_start,
                beta_end,
                timesteps,
                dtype=np.float64,  # Using float64 for better precision
            )
            self.num_timesteps = int(timesteps)
            times = np.arange(0, timesteps)
            alphas = 1.0 - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        
        # Define the cosine variance schedule
        elif schedule=='cosine':
            times = np.arange(0, timesteps)
            alphas_cumprod = cosineFunc(times, timesteps)/cosineFunc(np.zeros_like(times), timesteps)
            alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
            alphas = alphas_cumprod/alphas_cumprod_prev
            betas = 1.0 - alphas
        
        # Define the quadratic variance schedule
        elif schedule=='quadratic':
            order = 2
            times = np.arange(0, timesteps)
            betas = np.power(np.linspace(beta_end**(1. / order), beta_start**(1. / order), timesteps), order)[::-1]
            alphas = 1.0 - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        
        else:
            raise Exception('wrong schedule!')

        # set parameters
        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)
        self.sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=tf.float32)
        self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32)
        
        # noise parameter plot
        plot_noise_parameters(times, alphas_cumprod, betas, alphas, schedule=schedule, sampler=False)

    
    def _extract(self, a, t, x_shape):
        ''''Extract some coefficients at specified timesteps'''
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        reshaped_coef = tf.reshape(out, [batch_size, 1, 1, 1])
        return reshaped_coef

    def q_sample(self, x_start, t, noise):
        '''Diffuse the data'''
        x_start_shape = tf.shape(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start)) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
            * noise)