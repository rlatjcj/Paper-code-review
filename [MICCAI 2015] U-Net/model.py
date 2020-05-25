import keras
from keras.layers import ZeroPadding2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Concatenate
from keras.layers import Conv2DTranspose
from keras.layers import Reshape
from keras.models import Model

def UnetConv2D(input, outdim, is_batchnorm=False):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def Unet(input_shape=(None, None, 3), classes=21):
    img_input = Input(shape=input_shape)

    # Block 1
    c1 = UnetConv2D(img_input, 64)
    p1 = MaxPooling2D(pool_size=(2, 2), strides=2)(c1)
    
    # Block 2
    c2 = UnetConv2D(p1, 128)
    p2 = MaxPooling2D(pool_size=(2, 2), strides=2)(c2)

    # Block 3
    c3 = UnetConv2D(p2, 256)
    p3 = MaxPooling2D(pool_size=(2, 2), strides=2)(c3)

    # Block 4
    c4 = UnetConv2D(p3, 512)
    p4 = MaxPooling2D(pool_size=(2, 2), strides=2)(c4)

    # Block 5
    c5 = UnetConv2D(p4, 1024)

    # Block 6
    u6 = Conv2DTranspose(512, (2, 2), activation='relu', strides=(2, 2), padding='same')(c5)
    u6 = Concatenate()([u6, c4])
    c6 = UnetConv2D(u6, 512)

    # Block 7
    u7 = Conv2DTranspose(256, (2, 2), activation='relu', strides=(2, 2), padding='same')(c6)
    u7 = Concatenate()([u7, c3])
    c7 = UnetConv2D(u7, 256)

    # Block 8
    u8 = Conv2DTranspose(128, (2, 2), activation='relu', strides=(2, 2), padding='same')(c7)
    u8 = Concatenate()([u8, c2])
    c8 = UnetConv2D(u8, 128)

    # Block 9
    u9 = Conv2DTranspose(64, (2, 2), activation='relu', strides=(2, 2), padding='same')(c8)
    u9 = Concatenate()([u9, c1])
    c9 = UnetConv2D(u9, 64)

    c10 = Conv2D(classes, (1, 1))(c9)
    r10 = Reshape((-1, classes))(c10)
    logits = Activation('softmax')(r10)

    model = Model(img_input, logits, name='Unet')

    return model

if __name__ == '__main__':
    model = Unet((256,256,3))
    model.summary()