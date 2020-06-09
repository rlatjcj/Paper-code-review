import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

##########################
# Stage Functions
##########################
def BasicBlock(args, inputs, features, strides, name=None, **kwargs):
    x = Conv2D(features, (3, 3), strides=strides, use_bias=False, padding='same', name=name+'_conv1')(inputs)
    x = BatchNormalization(name=name+'_norm1')(x)
    x = Activation('relu', name=name+'_acti1')(x)
    x = Conv2D(features, (3, 3), strides=1, use_bias=False, padding='same', name=name+'_conv2')(x)
    x = BatchNormalization(name=name+'_norm2')(x)
    x = Activation('relu', name=name+'_acti2')(x)
    return x

def ResBlock(args, inputs, features, strides, name=None, **kwargs):
    shortcut = Conv2D(features, (1, 1), strides=strides, use_bias=False, name=name+'_shorcut_conv')(inputs)
    shortcut = BatchNormalization(name=name+'_shortcut_norm')(shortcut)

    x = Conv2D(features, (3, 3), strides=strides, use_bias=False, padding='same', name=name+'_conv1')(inputs)
    x = BatchNormalization(name=name+'_norm1')(x)
    x = Activation('relu', name=name+'_acti1')(x)
    x = Conv2D(features, (3, 3), strides=1, use_bias=False, padding='same', name=name+'_conv2')(x)
    x = BatchNormalization(name=name+'_norm2')(x)
    x = Add(name=name+'_add')([x, shortcut])
    x = Activation('relu', name=name+'_acti2')(x)
    return x

def ResBottleneckBlock(args, inputs, features, strides, name=None, **kwargs):
    shortcut = Conv2D(features, (1, 1), strides=strides, use_bias=False, name=name+'_shorcut_conv')(inputs)
    shortcut = BatchNormalization(name=name+'_shortcut_norm')(shortcut)

    x = Conv2D(features*args.bottleneck_ratio, (1, 1), strides=1, use_bias=False, padding='same', name=name+'_conv1')(inputs)
    x = BatchNormalization(name=name+'_norm1')(x)
    x = Activation('relu', name=name+'_acti1')(x)

    assert features % args.group_width == 0, 'The number of features must be divisible by group_width.'
    channel_per_group = features // args.group_width
    group_list = []
    for g in range(args.group_width):
        x_g = Lambda(lambda z: z[...,g*channel_per_group:(g+1)*channel_per_group])(x)
        x_g = Conv2D(channel_per_group*args.bottleneck_ratio, (3, 3), strides=strides, use_bias=False, padding='same', name=name+'_groupconv{}'.format(g+1))(x_g)
        group_list.append(x_g)
    x = Concatenate(name=name+'_groupconcat')(group_list)
    x = BatchNormalization(name=name+'_norm2')(x)
    x = Activation('relu', name=name+'_acti2')(x)

    x = Conv2D(features, (1, 1), strides=1, use_bias=False, padding='same', name=name+'_conv3')(x)
    x = BatchNormalization(name=name+'_norm3')(x)
    x = Add(name=name+'_add')([x, shortcut])
    x = Activation('relu', name=name+'_acti3')(x)
    return x


##########################
# AnyNet
##########################
def AnyNet(args, name=None, **kwargs):
    img_inputs = Input(shape=(args.img_size, args.img_size, 3), name='main_input')

    ##########################
    # Stem
    ##########################
    if 'cifar' in args.dataset:
        if args.stem == 'simple':
            x = Conv2D(args.stem_out, (3, 3), use_bias=False, padding='same', name='stem_conv')(img_inputs)
            x = BatchNormalization(name='stem_norm')(x)
            x = Activation('relu', name='stem_acti')(x)
        else:
            raise ValueError()
    else:
        if args.stem == 'simple':
            x = Conv2D(args.stem_out, (3, 3), strides=(2, 2), use_bias=False, padding='same', name='stem_conv')(img_inputs)
            x = BatchNormalization(name='stem_norm')(x)
            x = Activation('relu', name='stem_acti')(x)
        elif args.stem == 'resnet':
            x = Conv2D(args.stem_out, (7, 7), strides=(2, 2), use_bias=False, padding='same', name='stem_conv')(img_inputs)
            x = BatchNormalization(name='stem_norm')(x)
            x = Activation('relu', name='stem_acti')(x)
            x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='stem_pool')(x)
        else:
            raise ValueError()


    ##########################
    # Stage
    ##########################
    stage_dict = {
        'basic'         : BasicBlock,
        'res'           : ResBlock,
        'resbottleneck' : ResBottleneckBlock,
    }
    for s in range(args.n_stage):
        for b in range(args.n_block[s]):
            strides = 2 if b == 0 else 1
            x = stage_dict[args.type_stage](
                args, x, args.n_channel[s], strides, name='stage{}_block{}'.format(s+1, b+1))

    ##########################
    # Head
    ##########################
    x = GlobalAveragePooling2D()(x)
    x = Dense(args.classes)(x)
    img_outputs = Activation('softmax' if args.classes > 1 else 'sigmoid')(x)

    model = Model(img_inputs, img_outputs, name=name)
    return model