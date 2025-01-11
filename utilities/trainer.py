"""
@author: Fahim Ahmed Zaman
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from utilities.gaussianBlock import GaussianDiffusion
from utilities.model import Denoiser
import numpy as np

#%% Step-2: Denoiser and Image Encoder
''' 
This module is used for training LDSeg

'''

#%% Augmentation
def data_flip(imbatch, rots):
    n = rots.shape[0]
    imaugs = []
    for i in range(n):
        if rots[i]==0:
            imaugs.append(imbatch[i])
        elif rots[i]==3:
            imaugs.append(np.flip(imbatch[i], axis=(0, 1)))
        else:
            imaugs.append(np.flip(imbatch[i], axis=rots[i]-1))
    return np.array(imaugs, dtype=np.float32)

def perturb_flip(img, rot):
    nimg = tf.py_function(data_flip, [img, rot], tf.float32)
    return nimg

#%% Losses
def dice_coef_score(y_true, y_pred, num_labels=2, epsilon=1e-6):
    y_true_f = tf.cast(tf.one_hot(tf.cast(y_true, dtype=tf.uint8), num_labels)[...,1:], dtype=tf.float32)
    y_pred_f = tf.cast(y_pred[...,1:], dtype=tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_score = (2. * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + epsilon)
    return dice_score

def dice_coef(y_true, y_pred, num_labels=2, epsilon=1e-6):
    y_true_f = tf.cast(tf.one_hot(tf.cast(y_true, dtype=tf.uint8), num_labels), dtype=tf.float32)
    y_pred_f = tf.cast(y_pred, dtype=tf.float32)
    num = 2. * tf.reduce_sum(y_true_f * y_pred_f)
    den = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + epsilon
    dice_score = num / den
    return dice_score

def CE_loss(y_true, y_pred):
    scce = keras.losses.SparseCategoricalCrossentropy()
    ce_loss = scce(y_true, y_pred)
    return ce_loss

def DSC_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def reconstructionLoss(y_true, y_pred, gamma=2):
    return CE_loss(y_true, y_pred) + gamma * DSC_loss(y_true, y_pred)

#%% LDSeg framework
class LDSeg(keras.Model):
    def __init__(self, network, ema_network, imgEncoder, encoder, decoder, 
                 timesteps, gdf_util, filepath, augmentation=False, ema=0.999, lamda=1):
        
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.imgEncoder = imgEncoder
        self.encoder = encoder
        self.decoder = decoder
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.filepath = filepath,
        self.aug = augmentation
        self.ema = ema
        self.lamda = lamda
        
        print(self.filepath)
        
        # model paths
        self.encoderPath = os.path.join(self.filepath[0], 'labelEncoder.hdf5')
        self.decoderPath = os.path.join(self.filepath[0], 'labelDecoder.hdf5')
        self.imgEncoderPath = os.path.join(self.filepath[0], 'imageEncoder.hdf5')
        self.denoiserPath = os.path.join(self.filepath[0], 'denoiser.hdf5')
        self.modelPaths = [self.encoderPath, self.decoderPath, self.imgEncoderPath, self.denoiserPath]
        print('\nModel paths')
        print('---------------------------')
        for i in self.modelPaths:
            print(i)
        print('---------------------------\n')
        
    def compile(self, optimizerMaskEC, optimizerMaskDC, optimizerImgEC, optimizerDen, 
                loss1, loss2, score, *args, **kwargs):
        
        super().compile(*args, **kwargs)
        self.optimizerMaskEC = optimizerMaskEC
        self.optimizerMaskDC = optimizerMaskDC
        self.optimizerImgEC = optimizerImgEC
        self.optimizerDen = optimizerDen
        self.loss1 = loss1
        self.loss2 = loss2
        self.score = score
        
    def train_step(self, data):
        # load data
        X, y = data
        bsize = tf.shape(X)[0]
        Bx, Hx, Wx, Cx = X.get_shape()
        
        # augmentation
        if self.aug==True:
            flips = tf.random.uniform(minval=0, maxval=4, shape=(bsize, 1), dtype=tf.int64)
            X = perturb_flip(X, flips)
            y = perturb_flip(y, flips)
        X = tf.ensure_shape(X, [None, Hx, Wx, Cx])
        y = tf.ensure_shape(y, [None, Hx, Wx])
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(bsize, 1), dtype=tf.int64)
        
        # trainer        
        with tf.GradientTape(persistent=True) as tape:
            y_prime = tf.expand_dims(y, axis=-1)
            z_l0 = self.encoder(y_prime, training=True)
            z_i = self.imgEncoder(X, training=True)
            
            epsilon = tf.repeat(tf.expand_dims(tf.random.normal(shape=tf.shape(z_l0[0]), dtype=z_l0.dtype), axis=0), bsize, axis=0)
            z_lt = self.gdf_util.q_sample(z_l0, t, epsilon)
            z_nt = self.network([z_lt, z_i, t], training=True)
            z_dn = z_lt-z_nt
            y_hat = self.decoder(z_dn, training=True)
            
            # loss and score calculation
            loss1 = self.loss1(y, y_hat)
            loss2 = self.loss2(epsilon, z_nt)
            loss = loss1 + self.lamda * loss2
            score = self.score(y, y_hat)

        # gradients
        gradientsDen = tape.gradient(loss, self.network.trainable_weights)
        gradientsMaskEC = tape.gradient(loss, self.encoder.trainable_weights)
        gradientsMaskDC = tape.gradient(loss, self.decoder.trainable_weights)
        gradientsImgEC = tape.gradient(loss, self.imgEncoder.trainable_weights)

        # update weights
        self.optimizerDen.apply_gradients(zip(gradientsDen, self.network.trainable_weights))
        self.optimizerMaskEC.apply_gradients(zip(gradientsMaskEC, self.encoder.trainable_weights))
        self.optimizerMaskDC.apply_gradients(zip(gradientsMaskDC, self.decoder.trainable_weights))
        self.optimizerImgEC.apply_gradients(zip(gradientsImgEC, self.imgEncoder.trainable_weights))

        # updates with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
        
        # delete the tape to avoid memory leak
        del tape

        # return loss values
        return {"Loss(L)": loss, "Loss(L1)": loss1, "Loss(L2)": loss2,  "DSC": score}
    
    def save_model(self, epoch, logs=None):
        # save after every 100 epoch
        if np.mod(epoch,100)==0:
            self.encoder.save_weights(self.modelPaths[0])
            self.decoder.save_weights(self.modelPaths[1])
            self.imgEncoder.save_weights(self.modelPaths[2])
            self.ema_network.save_weights(self.modelPaths[3])
            print('\n---------------------------')
            print('Model weights saved...')
            print('---------------------------\n')
            
#%% trainer
def train_models(images, labels, 
                 labelEncoder, labelDecoder, imageEncoder, denoiser,
                 epochs = 1500, batch_size = 4, lrate = 1e-3,
                 minbeta = 1e-4, maxbeta = 0.02, total_timesteps = 1000, scheduler='cosine',
                 modelDirectory = './savedModels'):
    
    # data loader
    tf.keras.backend.clear_session()
    train_loader = tf.data.Dataset.from_tensor_slices((images, labels))
    train_dataset = train_loader.batch(batch_size).prefetch(batch_size)
    
    # EMA model
    ema = Denoiser(labelEncoder.output.shape[1:], 
                   imageEncoder.output.shape[1:])
    denoiser.set_weights(ema.get_weights())
    
    # Gaussian block
    gdf_util = GaussianDiffusion(beta_start=minbeta, 
                                 beta_end=maxbeta, 
                                 timesteps=total_timesteps, 
                                 schedule='cosine')
    
    # model framework
    model = LDSeg(network = ema,
                  ema_network = denoiser,
                  imgEncoder = imageEncoder, 
                  encoder = labelEncoder, 
                  decoder = labelDecoder,
                  timesteps = total_timesteps, 
                  gdf_util = gdf_util,
                  augmentation = True,
                  filepath = modelDirectory)
                           
    
    # model compile
    model.compile(optimizerMaskEC = Adam(learning_rate=lrate),
                  optimizerMaskDC = Adam(learning_rate=lrate),
                  optimizerImgEC = Adam(learning_rate=lrate),
                  optimizerDen = Adam(learning_rate=lrate),
                  loss1 = reconstructionLoss,
                  loss2 = keras.losses.MeanSquaredError(),
                  score = dice_coef_score)
    
    # train model
    model.fit(train_dataset,
              batch_size = batch_size,
              epochs = epochs,
              verbose = 1,
              callbacks = [keras.callbacks.LambdaCallback(on_epoch_end=model.save_model)])