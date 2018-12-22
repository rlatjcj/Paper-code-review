import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM

from keras.utils import plot_model

input_shape = (299, 299, 3)

def SeparableConvBlock(x, filters, strides, flow_name, num_block, num_conv):
    '''
    Separable Convolution Block used in MobileNet
    '''
    
    # Depwise Convolution
    x = KL.DepthwiseConv2D((3, 3), strides=(strides, strides), use_bias=False, padding='same', 
                           name='{}_block{}_depthwiseconv{}'.format(flow_name, num_block, num_conv))(x)
    x = KL.BatchNormalization(name='{}_block{}_BN{}_1'.format(flow_name, num_block, num_conv))(x)
    x = KL.Activation(activation='relu', name='{}_block{}_Relu{}_1'.format(flow_name, num_block, num_conv))(x)

    # Pointwise Convolution
    x = KL.Conv2D(filters, (1, 1), use_bias=False, padding='same',
                  name='{}_block{}_conv{}'.format(flow_name, num_block, num_conv))(x)
    x = KL.BatchNormalization(name='{}_block{}_BN{}_2'.format(flow_name, num_block, num_conv))(x)
    x = KL.Activation(activation='relu', name='{}_block{}_Relu{}_2'.format(flow_name, num_block, num_conv))(x)

    return x

def xception_block(x, filters, strides, flow_name, num_block):
    '''
    Xception Block in DeepLabv3+
    '''

    if flow_name in ['entry', 'exit']:
        residual = KL.Conv2D(filters[1], (1, 1), strides=(2, 2), use_bias=False, padding='same', name='{}_residual{}_conv'.format(flow_name, num_block))(x)
        residual = KL.BatchNormalization(name='{}_residual{}_BN'.format(flow_name, num_block))(residual)

    for i in range(3):
        x = SeparableConvBlock(x, filters[i],
                               strides=strides if i == 2 else 1,
                               flow_name=flow_name,
                               num_block=num_block,
                               num_conv=i+1)
    
    if flow_name in ['entry', 'exit']:
        x = KL.Add()([x, residual])

    return x


def DeepLabv3plus():
    img_input = KL.Input(shape=input_shape)
    s = KL.Lambda(lambda x: x / 255)(img_input)

    '''
    Encoder Module using Modified Xception Baseline
    '''
    # Entry flow
    x = KL.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding='same', name='entry_conv1')(s)
    x = KL.BatchNormalization(name='entry_conv1_BN')(x)
    x = KL.Activation(activation='relu', name='entry_conv1_Relu')(x)

    x = KL.Conv2D(64, (3, 3), use_bias=False, padding='same', name='entry_conv2')(x)
    x = KL.BatchNormalization(name='entry_conv2_BN')(x)
    x = KL.Activation(activation='relu', name='entry_conv2_Relu')(x)

    x = xception_block(x, [128, 128, 128], 2, 'entry', 1)
    x = xception_block(x, [256, 256, 256], 2, 'entry', 2)
    x = xception_block(x, [728, 728, 728], 2, 'entry', 3)

    # Middle flow
    for i in range(16):
        x = xception_block(x, [728, 728, 728], 1, 'middle', i+1)

    # Exit flow
    x = xception_block(x, [728, 1024, 1024], 2, 'exit', 1)
    
    x = xception_block(x, [1536, 1536, 2048], 1, 'last', 1)

    '''
    Encoder Module using Atrous Spatial Pyramid Pooling(ASPP) in DeepLabv3
    '''


    '''
    Decoder Module for Semantic Segmentation
    '''


    model = KM.Model(img_input, x)
    
    return model

if __name__ == '__main__':
    model = DeepLabv3plus()
    plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=False)
    model.summary()