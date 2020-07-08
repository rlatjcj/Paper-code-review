import keras
import keras.backend as K
from keras.layers import Input
from keras.layers import Conv3D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling3D
from keras.layers import Conv3DTranspose
from keras.layers import Concatenate
from keras.models import Model

def UnetConv3D(input, outdim1, outdim2, is_batchnorm=False):
    x = Conv3D(outdim1, (3, 3, 3), strides=(1, 1, 1), padding='same')(input)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(outdim2, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def Unet_3D(input_shape=(None, None, None, 3), classes=3):
    """3D U-Net
    
    Keyword Arguments:
        input_shape {tuple} -- [Set input shape you will use] (default: {(None, None, None, 3)})
        classes {int} -- [Set classes you will use] (default: {3})
    """

    img_input = Input(shape=input_shape)

    # Block 1
    c1 = UnetConv3D(img_input, 32, 64, True)
    p1 = MaxPooling3D()(c1)

    # Block 2
    c2 = UnetConv3D(p1, 64, 128, True)
    p2 = MaxPooling3D()(c2)

    # Block 3
    c3 = UnetConv3D(p2, 128, 256, True)
    p3 = MaxPooling3D()(c3)

    # Block 4
    c4 = UnetConv3D(p3, 256, 512, True)

    # Block 5
    u5 = Conv3DTranspose(512, (2, 2, 2), activation='relu', strides=(2, 2, 2), padding='same')(c4)
    u5 = Concatenate()([u5, c3])
    c5 = UnetConv3D(u5, 256, 256, True)

    # Block 6
    u6 = Conv3DTranspose(256, (2, 2, 2), activation='relu', strides=(2, 2, 2), padding='same')(c5)
    u6 = Concatenate()([u6, c2])
    c6 = UnetConv3D(u6, 128, 128, True)

    # Block 7
    u7 = Conv3DTranspose(128, (2, 2, 2), activation='relu', strides=(2, 2, 2), padding='same')(c6)
    u7 = Concatenate()([u7, c1])
    c7 = UnetConv3D(u7, 64, 64, True)

    logits = Conv3D(classes, (1, 1, 1), strides=(1, 1, 1), padding='same')(c7)

    # weighted softmax function

    model = Model(img_input, logits, name="Unet_3D")

    return model


if __name__ == "__main__":
    model = Unet_3D(input_shape=(None, None, None, 3))
    model.summary()