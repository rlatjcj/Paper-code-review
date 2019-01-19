import keras
import keras.backend as K
from keras.layers import Layer
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Concatenate
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
from keras.models import Model

from keras.utils.vis_utils import plot_model

class LRN(Layer):
    '''Local Response Normalization

    Reference:
    - AlexNet
        A. Krizhevsky, I. Sutskever and G. E. Hinton
        “ImageNet Classification with Deep Convolutional Neural Networks”,
        in Neural Information Processing Systems, 2012. 
    '''
    def __init__(self, batch=None, k=2, n=5, alpha=0.0001, beta=0.75, **kwargs):
        self.batch = batch
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta
        super(LRN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.b, self.h, self.w, self.ch = input_shape

    def call(self, x):
        x_sqr = K.square(x)
        half_n = self.n // 2
        real_inputs = K.zeros(shape=(self.batch, self.h, self.w, self.ch + half_n * 2), dtype=x.dtype)
        x_sqr = real_inputs[:,:,:,half_n:half_n+self.ch] + x_sqr

        scale = self.k
        norm_alpha = self.alpha / self.n
        for i in range(self.n):
            scale += norm_alpha * real_inputs[:,:,:,i:i+self.ch]

        output = K.pow(scale, self.beta)
        return output
        
    def compute_output_shape(self, input_shape):
        return input_shape

def Inception_Block(x, IB1, IB2_1, IB2_2, IB3_1, IB3_2, IB4):
    IB1_1x1_1s = Conv2D(filters=IB1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    IB2_1x1_1s = Conv2D(filters=IB2_1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    IB2_3x3_1s = Conv2D(filters=IB2_2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(IB2_1x1_1s)

    IB3_1x1_1s = Conv2D(filters=IB3_1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    IB3_3x3_1s = Conv2D(filters=IB3_2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(IB3_1x1_1s)

    IB4_3x3_1s = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    IB4_1x1_1s = Conv2D(filters=IB4, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(IB4_3x3_1s)

    IB = Concatenate()([IB1_1x1_1s, IB2_3x3_1s, IB3_3x3_1s, IB4_1x1_1s])
    return IB

def Inception_v1(input_shape=(None, None, 3), classes=1000, batch_size=1):
    img_input = Input(shape=input_shape)

    conv1_7x7_s2 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(img_input)
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_7x7_s2)
    pool1_norm = LRN(batch_size)(pool1_3x3_s2)

    conv2_1x1_reduce = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool1_norm)
    conv2_3x3_s1 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1x1_reduce)
    conv2_norm = LRN(batch_size)(conv2_3x3_s1)
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2_norm)

    inception_3a = Inception_Block(pool2_3x3_s2, 64, 96, 128, 16, 32, 32)
    inception_3b = Inception_Block(inception_3a, 128, 128, 192, 32, 96, 64)
    pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inception_3b)

    inception_4a = Inception_Block(pool3_3x3_s2, 192, 96, 208, 16, 48, 64)
    pool4a_5x5_s3 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(inception_4a)
    conv4a_1x1_s1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool4a_5x5_s3)
    FC1 = Flatten()(conv4a_1x1_s1)
    FC1_dense1 = Dense(units=1024)(FC1)
    FC1_dropout1 = Dropout(rate=0.7)(FC1_dense1)
    FC1_dense2 = Dense(units=classes)(FC1_dropout1)
    FC1_activation = Activation(activation='softmax')(FC1_dense2)

    inception_4b = Inception_Block(inception_4a, 160, 112, 224, 24, 64, 64)
    inception_4c = Inception_Block(inception_4b, 128, 128, 256, 24, 64, 64)
    inception_4d = Inception_Block(inception_4c, 112, 144, 288, 32, 64, 64)
    pool4d_5x5_s3 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(inception_4d)
    conv4d_1x1_s1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool4d_5x5_s3)
    FC2 = Flatten()(conv4d_1x1_s1)
    FC2_dense1 = Dense(units=1024)(FC2)
    FC2_dropout1 = Dropout(rate=0.7)(FC2_dense1)
    FC2_dense2 = Dense(units=classes)(FC2_dropout1)
    FC2_activation = Activation(activation='softmax')(FC2_dense2)

    inception_4e = Inception_Block(inception_4d, 256, 160, 320, 32, 128, 128)
    pool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inception_4e)

    inception_5a = Inception_Block(pool4_3x3_s2, 256, 160, 320, 32, 128, 128)
    inception_5b = Inception_Block(inception_5a, 384, 192, 384, 48, 128, 128)
    pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(inception_5b)

    FC3 = Flatten()(pool5_7x7_s1)
    FC3_dropout = Dropout(rate=0.4)(FC3)
    FC3_dense = Dense(units=classes)(FC3_dropout)
    FC3_activation = Activation(activation='softmax')(FC3_dense)
       

    model = Model(img_input, [FC1_activation, FC2_activation, FC3_activation], name='Inception_v1')

    return model

if __name__ == '__main__':
    model = Inception_v1((224, 224, 3), 1000)
    model.summary()
    plot_model(model, './Inception_v1.png', show_shapes=True)