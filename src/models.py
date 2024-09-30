from keras.models import Model
import tensorflow as tf
from keras import layers, models
from functools import partial


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


def build_resnet16(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, kernel_size=(3, 3), padding='same')(inputs)
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

    x = layers.Dense(4, activation='sigmoid')(x) 

    model = models.Model(inputs, x)
    return model

