"""
Model architectures for Active Learning trails.
"""

from keras.models import Model
import tensorflow as tf
from keras import layers, models
from functools import partial
import numpy as np


def residual_block(x, filters, down_sample=False):
    res = x # residual shortcut
    strides = [2, 1] if down_sample else [1, 1]
    default_conv2 = partial(layers.Conv2D, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')
    
    # first conv
    x = default_conv2(filters, strides=strides[0])(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # second conv
    x = default_conv2(filters, strides=strides[1])(x)
    x = layers.BatchNormalization()(x)
    
    # adjust the shortcut if downsampling
    if down_sample:
        res = default_conv2(filters, kernel_size=(1, 1), strides=2)(res)
        res = layers.BatchNormalization()(res)

    x = layers.Add()([x, res])
    x = layers.ReLU()(x)
    return x


def build_resnet16(input_shape, n_classes=5, multilabel=False):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, kernel_size=(3, 3), padding='same', 
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = residual_block(x, 16, down_sample=True)
    x = residual_block(x, 16)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 32, down_sample=True)
    x = residual_block(x, 32)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 64, down_sample=True)
    x = residual_block(x, 64)

    x = layers.GlobalAveragePooling2D()(x)

    act = 'sigmoid' if multilabel else 'softmax'
    x = layers.Dense(n_classes, activation=act)(x) 

    model = models.Model(inputs, x)
    return model


def build_compile_resnet16(input_shape=(40, 107, 1), n_classes=5, multilabel=False):
    import keras_cv
    model = build_resnet16(input_shape) 
    model.compile(
        optimizer='adam',
        loss=keras_cv.losses.FocalLoss(alpha=0.25, gamma=2)
    )
    return model