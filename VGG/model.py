# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten


def VGG(input_shape = (224, 224, 3), classes = 1000, type='D'):
    img_input = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(img_input)
    if type in ['B', 'C', 'D', 'E']:
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    if type in ['B', 'C', 'D', 'E']:
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    if type == 'C':
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    elif type == 'D':
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    elif type == 'E':
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    if type == 'C':
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    elif type == 'D':
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    elif type == 'E':
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    if type == 'C':
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    elif type == 'D':
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    elif type == 'E':
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    flat_6 = Flatten()(x)
    fc6_1 = Dense(4096, activation='relu')(flat_6)
    fc6_1_do = Dropout(0.5)(fc6_1)
    fc6_2 = Dense(4096, activation='relu')(fc6_1_do)
    fc6_2_do = Dropout(0.5)(fc6_2)
    logits = Dense(classes, activation='softmax')(fc6_2_do)

    model = Model(img_input, logits, name='VGG')
    
    return model

if __name__ == '__main__':
    # from keras.utils.vis_utils import plot_model
    model = VGG()
    model.summary()
    # plot_model(model, to_file='./model.png')