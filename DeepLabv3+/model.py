# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import AveragePooling2D
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import ZeroPadding2D
from keras.layers import Add
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Reshape
from keras.engine.topology import Layer


class BilinearUpSampling(Layer):
    def __init__(self, size, name, **kwargs):
        super(BilinearUpSampling, self).__init__(**kwargs)
        self.size = size
        self.name = name

    def call(self, x):
        return tf.image.resize_bilinear(x, size=self.size, align_corners=True, name=self.name)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.size[0], self.size[1], input_shape[3])

def SeparableConvBlock(x, filters, strides=1, rate=1, prefix=None, last_activation=False):
    '''
    Separable Convolution Block used in MobileNet and Xception
    '''
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(padding=(1, 1))(x)
        padding='valid'

    if not last_activation:
        x = Activation(activation='relu', name=prefix+'_early_act')(x)

    # Depwise Convolution
    x = DepthwiseConv2D((3, 3), strides=(strides, strides), use_bias=False, padding=padding, 
                           dilation_rate=(rate, rate),
                           name=prefix+'_depthwise_conv')(x)
    x = BatchNormalization(name=prefix+'_depthwise_bn')(x)
    x = Activation(activation='relu', name=prefix+'_depthwise_act')(x)

    # Pointwise Convolution
    x = Conv2D(filters, (1, 1), use_bias=False, padding='same',
                  name=prefix+'_pointwise_conv')(x)
    x = BatchNormalization(name=prefix+'_pointwise_bn')(x)
    if last_activation:
        x = Activation(activation='relu', name=prefix+'_pointwise_act')(x)

    return x

def xception_block(inputs, filters, strides, rate=1, prefix=None, skip_connection=False, mode='conv', last_activation=False):
    '''
    Xception Block in DeepLabv3+
    '''

    x = inputs
    for i in range(3):
        x = SeparableConvBlock(x, filters[i],
                               strides=strides if i == 2 else 1,
                               rate=rate,
                               prefix=prefix+'_seprable'+str(i+1),
                               last_activation=last_activation)

        if i == 1 and skip_connection:
            skip = x
    
    if mode == 'conv':
        residual = Conv2D(filters[1], (1, 1), strides=(strides, strides), use_bias=False, padding='same', name=prefix+'_residual_conv')(inputs)
        residual = BatchNormalization(name=prefix+'_residual_bn')(residual)
        x = Add()([x, residual])
    elif mode == 'sum':
        x = Add()([x, inputs])

    if skip_connection:
        return x, skip
    else:
        return x


def DeepLabv3plus(input_shape = (513, 513, 3),
                  classes = 21):
    img_input = Input(shape=input_shape)

    '''
    Encoder Module using Modified Xception Baseline
    '''
    # Entry flow
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation(activation='relu', name='block1_conv1_act')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation(activation='relu', name='block1_conv2_act')(x)

    x = xception_block(x, [128, 128, 128], 2, 1, 'block2', mode='conv')
    x, skip = xception_block(x, [256, 256, 256], 2, 1, 'block3', True, mode='conv')
    x = xception_block(x, [728, 728, 728], 2, 1, 'block4', mode='conv')

    # Middle flow
    for i in range(16):
        prefix = 'block'+str(i+5)
        x = xception_block(x, [728, 728, 728], 1, 1, prefix, mode='sum')

    # Exit flow
    x = xception_block(x, [728, 1024, 1024], 1, 1, 'block21', mode='conv')
    x = xception_block(x, [1536, 1536, 2048], 1, 2, 'block22', mode='none', last_activation=True)

    '''
    Encoder Module using Atrous Spatial Pyramid Pooling(ASPP) in DeepLabv3 and ParseNet
    '''
    b0 = Conv2D(256, (1, 1), use_bias=False, padding='same', name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN')(b0)
    b0 = Activation(activation='relu', name='aspp0_Relu')(b0)
    
    b1 = SeparableConvBlock(x, 256, prefix='aspp1', rate=6, last_activation=True)
    b2 = SeparableConvBlock(x, 256, prefix='aspp2', rate=12, last_activation=True)
    b3 = SeparableConvBlock(x, 256, prefix='aspp3', rate=18, last_activation=True)
    
    b4 = AveragePooling2D(pool_size=(K.int_shape(x)[1], K.int_shape(x)[2]), name='image_pooling')(x)
    b4 = Conv2D(256, (1, 1), use_bias=False, padding='same', name='image_pooling_conv')(b4)
    b4 = BatchNormalization(name='image_pooling_bn')(b4)
    b4 = Activation(activation='relu', name='image_pooling_act')(b4)
    b4 = BilinearUpSampling(size=(K.int_shape(x)[1], K.int_shape(x)[2]), name='image_pooling_unpool')(b4)

    x = Concatenate()([b0, b1, b2, b3, b4])
    x = Conv2D(256, (1, 1), use_bias=False, padding='same', name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_bn')(x)
    x = Activation(activation='relu', name='concat_projection_act')(x)

    '''
    Decoder Module for Semantic Segmentation
    '''
    # Skip Connection
    x = BilinearUpSampling(size=(K.int_shape(skip)[1], K.int_shape(skip)[2]), name='first_upsampling')(x)
    
    skip = Conv2D(48, (1, 1), use_bias=False, padding='same', name='feature_projection0')(skip)
    skip = BatchNormalization(name='feature_projection0_bn')(skip)
    skip = Activation(activation='relu', name='feature_projection0_act')(skip)
    x = Concatenate()([x, skip])
    x = SeparableConvBlock(x ,256, prefix='decoder_conv0', last_activation=True)
    x = SeparableConvBlock(x ,256, prefix='decoder_conv1', last_activation=True)
    x = Conv2D(classes, (1, 1), padding='same', name='logits')(x)
    x = BilinearUpSampling(size=(input_shape[0], input_shape[1]), name='second_upsampling')(x)

    x = Reshape((-1, classes))(x)
    logits = Activation(activation='softmax')(x)

    model = Model(img_input, logits)
    
    return model

if __name__ == '__main__':
    from keras.utils import plot_model

    model = DeepLabv3plus()
    model.summary()
    plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True)