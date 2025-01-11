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
        x = layers.Conv2D(width, kernel_size=3, strides=2, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply

#%% Convolutional blocks
def res_conv_block(x, 
                   filter_size, 
                   size, 
                   dropout, 
                   batch_norm=False, 
                   activation='relu', 
                   layer_name=False):
    '''
    Residual convolutional layer.
    
    '''
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same', kernel_initializer='he_uniform')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=-1)(conv)
    if activation=='swish':
        conv = keras.activations.swish(conv)
    else:
        conv = layers.Activation('relu')(conv)
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same', kernel_initializer='he_uniform')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=-1)(conv)
    #conv = layers.Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=-1)(shortcut)

    res_path = layers.add([shortcut, conv])
    if activation=='swish':
        if layer_name==False:
            res_path = keras.activations.swish(res_path)
        else:
            res_path = layers.Activation(keras.activations.swish, name=layer_name)(res_path)
    else:
        if layer_name==False:
            res_path = layers.Activation('relu')(res_path)
        else:
            res_path = layers.Activation('relu', name=layer_name)(res_path)
    return res_path

def conv_block(x, 
               filter_size, 
               kernel_size, 
               activation_fn, 
               groups=4, 
               dropout_rate=True):
    
    residual = layers.Conv2D(filter_size, kernel_size=1, padding='same', kernel_initializer=kernel_init(1.0))(x)
    x = activation_fn(x)
    x = layers.Conv2D(filter_size, kernel_size=kernel_size, padding="same", kernel_initializer=kernel_init(1.0))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = activation_fn(x)
    x = layers.Conv2D(filter_size, kernel_size=kernel_size, padding="same", kernel_initializer=kernel_init(0.0))(x)
    x = layers.Add()([x, residual])
    x = activation_fn(x)
    x = DownSample(filter_size)(x)
    x = layers.GroupNormalization(groups=groups)(x)
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
            residual = layers.Conv2D(width, kernel_size=1, kernel_initializer=kernel_init(1.0))(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[:, None, None, :]

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)

        x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0))(x)
        x = layers.Add()([x, residual])
        return x

    return apply

#%% Attention mechanisms
class MultiHeadAttentionBlock(layers.Layer):
    """Applies multi-head self-attention.

    Args:
        units: Number of units in the dense layers.
        num_heads: Number of attention heads.
        groups: Number of groups for GroupNormalization layer.
    """
    def __init__(self, units, num_heads=8, groups=8, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.groups = groups

        # Define the GroupNormalization and Dense layers for query, key, value, and projection
        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def split_heads(self, x, batch_size):
        """Splits the last dimension into (num_heads, depth) and transposes the result
        to shape (batch_size, num_heads, height, width, depth).
        """
        depth = self.units // self.num_heads
        x = tf.reshape(x, (batch_size, -1, self.num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, tokens, depth)

    def call(self, inputs):
        shape = tf.shape(inputs)
        batch_size, height, width, _ = shape[0], shape[1], shape[2], shape[3]
        num_tokens = height * width
        scale = tf.cast(self.units // self.num_heads, tf.float32) ** (-0.5)
    
        # Normalize inputs
        inputs_norm = self.norm(inputs)
    
        # Compute Q, K, V
        q = self.query(inputs_norm)  # (B, H, W, units)
        k = self.key(inputs_norm)    # (B, H, W, units)
        v = self.value(inputs_norm)  # (B, H, W, units)
    
        # Reshape and split into heads
        q = tf.reshape(q, (batch_size, num_tokens, self.units))  # (B, H*W, units)
        k = tf.reshape(k, (batch_size, num_tokens, self.units))
        v = tf.reshape(v, (batch_size, num_tokens, self.units))
    
        q = self.split_heads(q, batch_size)  # (B, num_heads, H*W, depth)
        k = self.split_heads(k, batch_size)  # (B, num_heads, H*W, depth)
        v = self.split_heads(v, batch_size)  # (B, num_heads, H*W, depth)
    
        # Compute attention scores
        attn_score = tf.einsum("bhid,bhjd->bhij", q, k) * scale  # (B, num_heads, H*W, H*W)
        attn_score = tf.nn.softmax(attn_score, axis=-1)  # Softmax over the last dimension (H*W)
    
        # Apply attention to values
        attn_output = tf.einsum("bhij,bhjd->bhid", attn_score, v)  # (B, num_heads, H*W, depth)
    
        # Concatenate heads and project back to the original dimension
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])  # (B, H*W, num_heads, depth)
        attn_output = tf.reshape(attn_output, (batch_size, height, width, self.units))  # (B, H, W, units)
    
        # Final projection layer
        proj_output = self.proj(attn_output)
    
        # Ensure residual connection has matching shape
        if inputs.shape[-1] == proj_output.shape[-1]:
            return inputs + proj_output
        else:
            raise ValueError("Shape mismatch between input and projected output for residual connection")
            
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
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        
        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])
        
        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])
        
        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
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
            FILTER_NUM = 16, 
            activation = 'swish'):
                  
    inputs = layers.Input(input_shape, dtype=tf.float32, name='encoder_input')
    
    for i in range(len(elayers)):
        if i==0:
            x = res_conv_block(inputs, 
                               FILTER_SIZE, 
                               elayers[i]*FILTER_NUM, 
                               dropout_rate, 
                               batch_norm, 
                               activation=activation)
        else:
            x = res_conv_block(x, 
                               FILTER_SIZE, 
                               elayers[i]*FILTER_NUM, 
                               dropout_rate, 
                               batch_norm, 
                               activation=activation)
        if i!=len(elayers)-1:
            x = layers.MaxPooling2D(pool_size=(2, 2)) (x)
                
    x = layers.Conv2D(1, kernel_size=(1, 1), padding='same') (x)
    encoded = layers.LayerNormalization(axis=(1, 2, 3), center=True, scale=True, name='encoder_output')(x)
    model = models.Model(inputs, encoded, name="Label-Encoder")
    return model

def decoder(input_shape,
            dlayers = [2, 4, 4, 2],
            dropout_rate = 0.2, 
            batch_norm = True, 
            FILTER_SIZE = 3, 
            FILTER_NUM = 16, 
            activation = 'swish',
            num_classes = 2):
    
    inputs = layers.Input(input_shape, dtype=tf.float32, name='decoder_input')
    
    for i in range(len(dlayers)):
        if i==0:
            x = layers.Conv2DTranspose(filters = dlayers[i]*FILTER_NUM, 
                                       kernel_size=(3, 3), 
                                       padding="same", 
                                       strides=2) (inputs)
        else:
            x = layers.Conv2DTranspose(filters = dlayers[i]*FILTER_NUM, 
                                       kernel_size=(3, 3), 
                                       padding="same", 
                                       strides=2) (x)
        x = res_conv_block(x, 
                           FILTER_SIZE, 
                           dlayers[i]*FILTER_NUM, 
                           dropout_rate, 
                           batch_norm, 
                           activation=activation)
        
    x = keras.activations.swish(x)
    x = layers.Conv2D(num_classes, kernel_size=(1, 1)) (x)
    x = layers.BatchNormalization(axis=-1) (x)
    decoded = layers.Softmax(axis=-1, name='decoder_output')(x)
    model = models.Model(inputs, decoded, name="Label-Decoder")
    return model

def ImgEncoder(input_shape,
               filter_size=16,
               kernel_size=3, 
               dropout=0.2, 
               groups=4,
               channels = 1,
               activation=keras.activations.swish):  
    
    inputs = layers.Input(input_shape, dtype=tf.float32, name='image_input')
    
    x = layers.Conv2D(filter_size, kernel_size=3, padding='same', kernel_initializer=kernel_init(1.0))(inputs)
    x = conv_block(x, 2*filter_size, kernel_size, dropout_rate=dropout, activation_fn=activation)
    x = conv_block(x, 4*filter_size, kernel_size, dropout_rate=dropout, activation_fn=activation)
    x = conv_block(x, 4*filter_size, kernel_size, dropout_rate=dropout, activation_fn=activation)
    x = MultiHeadAttentionBlock(4*filter_size)(x)
    x = conv_block(x, 2*filter_size, kernel_size, dropout_rate=dropout, activation_fn=activation)
    x = MultiHeadAttentionBlock(2*filter_size)(x)
    x = layers.Conv2D(channels, kernel_size=1, padding="same", kernel_initializer=kernel_init(0.0))(x)
    x = MultiHeadAttentionBlock(channels, num_heads=channels, groups=1)(x)
    x = activation(x)
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
    
    x = layers.Conv2D(first_conv_channels, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_init(1.0))(x)
    
    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    
    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(widths[i], 
                              groups=norm_groups, 
                              activation_fn=activation_fn)([x, temb])
            
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)
        

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    
    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(widths[i], 
                              groups=norm_groups, 
                              activation_fn=activation_fn)([x, temb])
            
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    outputs = layers.Conv2D(channels, (3, 3), padding="same", name='denoiser_output')(x)
    return keras.Model(inputs, outputs, name="Denoiser")

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
        modelNames = ['labelEncoder.hdf5', 'labelDecoder.hdf5', 'imageEncoder.hdf5', 'denoiser.hdf5']
        savedmodels = os.listdir(filepath)
        if all(item in savedmodels for item in modelNames):
            for modl, path in zip(models, modelNames):
                modl.load_weights(os.path.join(filepath, path))
                print(f'\n{os.path.splitext(path)[0]} weights loaded...')
        else:
            print('\nModel weights are unavailable. Please train LDSeg...\n')
    return models