"""
@author: Fahim Zaman

email: fahim-zaman@uiowa.edu
"""

from utilities import dataLoader3D, model3D, trainer3D, sampler3D, misc
from configparser import ConfigParser

#%% Load Configurations
config = ConfigParser()
config.read('cfg.ini')

# GPU = config.getboolean('GPU', 'Multiple')
# misc.setGPU(GPU)

MINBETA = config.getfloat('NoiseScheduler', 'beta_0')
MAXBETA = config.getfloat('NoiseScheduler', 'beta_T')
SCHEDULER = config.get('NoiseScheduler', 'Scheduler')
TOTAL_STEPS = config.getint('NoiseScheduler', 'Timesteps')

MODEL_TRAINING = config.getboolean('Parameters', 'Training')
CHECKPOINT = config.getboolean('Parameters', 'Checkpoint')
BATCH_SIZE = config.getint('Parameters', 'BatchSize')
EPOCH = config.getint('Parameters', 'Epoch')
LEARNING_RATE = config.getfloat('Parameters', 'LearningRate')

SAMPLING_STEPS = config.getint('Sampler', 'Sampling_Steps')
SAMPLER = config.get('Sampler', 'Sampler')
SIGMA = config.getfloat('Sampler', 'Sigma')

#%% Data Read
images, labels = dataLoader3D.dataRead()

#%% Load Model Components
components = model3D.loadModel(images, 
                               labels, 
                               loadCheckpoint = CHECKPOINT)                                 

labelEncoder, labelDecoder, imageEncoder, denoiser = components

#%% Model Training
if MODEL_TRAINING == True:
    trainer3D.train_models(images = images,
                           labels = labels,
                           labelEncoder = labelEncoder,
                           labelDecoder = labelDecoder,
                           imageEncoder = imageEncoder,
                           denoiser = denoiser,
                           minbeta = MINBETA,
                           maxbeta = MAXBETA,
                           total_timesteps = TOTAL_STEPS,
                           scheduler = SCHEDULER,
                           epochs = EPOCH,
                           batch_size = BATCH_SIZE,
                           lrate = LEARNING_RATE)                         
    
#%% Model inference
yhat = sampler3D.segment(data = images,
                         labelDecoder = labelDecoder, 
                         imageEncoder = imageEncoder, 
                         denoiser = denoiser, 
                         totalTimesteps = TOTAL_STEPS, 
                         samplingSteps = SAMPLING_STEPS, 
                         sampler = SAMPLER,
                         scheduler = SCHEDULER,
                         sigma = SIGMA, 
                         batch_size = BATCH_SIZE)                                                              

#%% Write Segmentations
dataLoader3D.segWrite(segmentation = yhat)