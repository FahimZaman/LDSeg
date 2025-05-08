"""
@author: Fahim Ahmed Zaman
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import math

#%%
''' 
This module is used for initializing the model components

'''

#%% Kernel initialization
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )

#%% Up and Down-sampling
def DownSample(width):
    def apply(x):
        x = layers.Conv3D(width, kernel_size=3, strides=2, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling3D(size=2)(x)
        x = layers.Conv3D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply

#%% Convolutional blocks
def res_conv_block(x, 
                   filter_size, 
                   size, 
                   dropout, 
                   batch_norm=False):
    '''
    Residual convolutional layer.
    
    '''
    conv = layers.Conv3D(size, (filter_size, filter_size, filter_size), padding='same', kernel_initializer='he_uniform')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=-1)(conv)
    conv = keras.activations.swish(conv)
    
    conv = layers.Conv3D(size, (filter_size, filter_size, filter_size), padding='same', kernel_initializer='he_uniform')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=-1)(conv)
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv3D(size, kernel_size=(1, 1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=-1)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = keras.activations.swish(res_path)
    return res_path

def conv_block(x, 
               filter_size, 
               kernel_size, 
               activation_fn, 
               dropout_rate=0.2, 
               num_channels=4):
    residual = layers.Conv3D(filter_size, kernel_size=1, padding='same', kernel_initializer=kernel_init(1.0))(x)
    x = activation_fn(x)
    x = layers.Conv3D(filter_size, kernel_size=kernel_size, padding="same", groups=num_channels, kernel_initializer=kernel_init(1.0))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = activation_fn(x)
    x = layers.Conv3D(filter_size, kernel_size=kernel_size, padding="same", groups=num_channels, kernel_initializer=kernel_init(0.0))(x)
    x = layers.Add()([x, residual])
    x = activation_fn(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = layers.GroupNormalization(groups=num_channels)(x)
    return x

def ResidualBlock(width, 
                  groups=8, 
                  activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[-1]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv3D(width, kernel_size=1, kernel_initializer=kernel_init(1.0))(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[:, None, None, None, :]

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv3D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)

        x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv3D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0))(x)
        x = layers.Add()([x, residual])
        return x

    return apply

#%% Attention mechanisms
class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        depth = tf.shape(inputs)[1]
        height = tf.shape(inputs)[2]
        width = tf.shape(inputs)[3]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        
        attn_score = tf.einsum("bdhwc, bDHWc->bdhwDHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, depth, height, width, depth * height * width])
        
        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, depth, height, width, depth, height, width])
        
        proj = tf.einsum("bdhwDHW,bDHWc->bdhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj

#%% Time embeddings
class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb
    
def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb
    return apply

#%% Model components
def encoder(input_shape,
            elayers = [1, 2, 4, 4, 2],
            dropout_rate = 0.2, 
            batch_norm = True, 
            FILTER_SIZE = 3, 
            FILTER_NUM = 16):
                  
    inputs = layers.Input(input_shape, dtype=tf.float32, name='encoder_input')
    
    for i in range(len(elayers)):
        if i==0:
            x = res_conv_block(inputs, 
                               FILTER_SIZE, 
                               elayers[i]*FILTER_NUM, 
                               dropout_rate, 
                               batch_norm)
        else:
            x = res_conv_block(x, 
                               FILTER_SIZE, 
                               elayers[i]*FILTER_NUM, 
                               dropout_rate, 
                               batch_norm)
        if i!=len(elayers)-1:
            x = DownSample(elayers[i]*FILTER_NUM) (x)
                
    x = layers.Conv3D(1, kernel_size=(2, 2, 2), padding='same') (x)
    encoded = layers.LayerNormalization(axis=(1, 2, 3, 4), center=True, scale=True, name='encoder_output')(x)
    model = models.Model(inputs, encoded, name="Mask-Encoder")
    return model

def decoder(input_shape,
            dlayers = [2, 4, 4, 2],
            dropout_rate = 0.2, 
            batch_norm = True, 
            FILTER_SIZE = 3, 
            FILTER_NUM = 16,
            num_channels = 3):
    
    inputs = layers.Input(input_shape, dtype=tf.float32, name='decoder_input')
    
    for i in range(len(dlayers)):
        if i==0:
            x = UpSample(dlayers[i]*FILTER_NUM) (inputs)
        else:
            x = UpSample(dlayers[i]*FILTER_NUM) (x)
        x = res_conv_block(x, 
                           FILTER_SIZE, 
                           dlayers[i]*FILTER_NUM, 
                           dropout_rate, 
                           batch_norm)
        
    x = keras.activations.swish(x)
    x = layers.Conv3D(num_channels, kernel_size=(1, 1, 1)) (x)
    x = layers.BatchNormalization(axis=-1) (x)
    decoded = layers.Softmax(axis=-1, name='decoder_output')(x)
    model = models.Model(inputs, decoded, name="Mask-Decoder")
    return model

def ImgEncoder(input_shape,
               ilayers = [2, 4, 4, 2],
               filter_size = 16,
               kernel_size = 3, 
               dropout = 0.2, 
               activation = keras.activations.swish):             
    
    inputs = layers.Input(input_shape, dtype=tf.float32, name='image_input')
    
    x = layers.Conv3D(filter_size, kernel_size=3, padding='same', kernel_initializer=kernel_init(1.0))(inputs)
    
    for i in range(len(ilayers)):
        x = conv_block(x, 
                       ilayers[i]*filter_size, 
                       kernel_size, 
                       dropout_rate=dropout, 
                       activation_fn=activation)
    
    x = layers.Conv3D(1, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0))(x)
    outputs = layers.BatchNormalization(axis=-1, name='embedding_output')(x)
    
    # Model 
    model = models.Model(inputs, outputs, name="Image-Encoder")
    return model

def Denoiser(input_shape_lv,
             input_shape_ie,
             first_conv_channels = 16,
             widths = [16, 32, 64],
             has_attention = [False, True, True],
             num_res_blocks=2,
             norm_groups=4,
             channels=1,
             interpolation="nearest",
             activation_fn=keras.activations.swish):
    
    lv_input = layers.Input(input_shape_lv, dtype=tf.float32)
    img_input = layers.Input(input_shape_ie, dtype=tf.float32)
    time_input = layers.Input((), dtype=tf.float32)
    
    inputs = [lv_input, img_input, time_input]
    
    x = layers.Concatenate(axis=-1, name='denoiser_input')([lv_input, img_input])
    
    x = layers.Conv3D(first_conv_channels, kernel_size=(3, 3, 3), padding="same", kernel_initializer=kernel_init(1.0))(x)
    
    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    
    skips = []

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)
            

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            # skips.append(x)
    
    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    
    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    outputs = layers.Conv3D(channels, (3, 3, 3), padding="same", name='denoiser_output')(x)
    return keras.Model(inputs, outputs, name="denoiser")

#%% load model components
def loadModel(images, labels, filepath='./savedModels', loadCheckpoint=True):
    tf.keras.backend.clear_session()
    labelEncoder = encoder(np.expand_dims(labels, axis=-1).shape[1:])
    labelDecoder = decoder(labelEncoder.output.shape[1:])
    imageEncoder = ImgEncoder(images.shape[1:])
    denoiser = Denoiser(labelEncoder.output.shape[1:], imageEncoder.output.shape[1:])
    
    # model components
    models = (labelEncoder, labelDecoder, imageEncoder, denoiser)
    
    # load checkpoints
    if loadCheckpoint == True:
        modelNames = ['labelEncoder3D.hdf5', 
                      'labelDecoder3D.hdf5', 
                      'imageEncoder3D.hdf5', 
                      'denoiser3D.hdf5']
        savedmodels = os.listdir(filepath)
        if all(item in savedmodels for item in modelNames):
            for modl, path in zip(models, modelNames):
                modl.load_weights(os.path.join(filepath, path))
                print(f'\n{os.path.splitext(path)[0]} weights loaded...')
        else:
            print('\nModel weights are unavailable. Please train LDSeg...\n')
    return models