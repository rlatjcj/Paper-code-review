import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

##########################
# Stage Functions
##########################
def BasicBlock(args, x, features, strides, name=None, **kwargs):
    x = Conv2D(features, 3, strides=strides, padding='same', use_bias=False, name=name+'_conv1')(x)
    x = BatchNormalization(name=name+'_norm1')(x)
    x = Activation('relu', name=name+'_acti1')(x)
    
    x = Conv2D(features, 3, strides=1, padding='same', use_bias=False, name=name+'_conv2')(x)
    x = BatchNormalization(name=name+'_norm2')(x)
    x = Activation('relu', name=name+'_acti2')(x)
    return x

def ResBlock(args, x, features, strides, name=None, **kwargs):
    shortcut = Conv2D(features, 1, strides=strides, padding='valid', use_bias=False, name=name+'_shorcut_conv')(x)
    shortcut = BatchNormalization(name=name+'_shortcut_norm')(shortcut)

    x = Conv2D(features, 3, strides=strides, padding='same', use_bias=False, name=name+'_conv1')(x)
    x = BatchNormalization(name=name+'_norm1')(x)
    x = Activation('relu', name=name+'_acti1')(x)

    x = Conv2D(features, 3, strides=1, padding='same', use_bias=False, name=name+'_conv2')(x)
    x = BatchNormalization(name=name+'_norm2')(x)
    x = Add(name=name+'_add')([x, shortcut])
    x = Activation('relu', name=name+'_acti2')(x)
    return x

def ResBottleneckBlock(args, x, features, strides, name=None, **kwargs):
    shortcut = Conv2D(features, 1, strides=strides, padding='valid', use_bias=False, name=name+'_shorcut_conv')(x)
    shortcut = BatchNormalization(name=name+'_shortcut_norm')(shortcut)

    x = Conv2D(features//args.bottleneck_ratio, 1, strides=1, padding='valid', use_bias=False, name=name+'_conv1')(x)
    x = BatchNormalization(name=name+'_norm1')(x)
    x = Activation('relu', name=name+'_acti1')(x)
    
    if args.group_width == 1:
        x = Conv2D(features//args.bottleneck_ratio, 3, strides=strides, padding='same', use_bias=False, name=name+'_conv2')(x)
    else:
        channel_per_group = (features//args.bottleneck_ratio) // args.group_width
        group_list = []
        for g in range(args.group_width):
            x_g = Lambda(lambda z: z[...,g*channel_per_group:(g+1)*channel_per_group])(x)
            x_g = Conv2D(channel_per_group, 3, strides=strides, padding='same', use_bias=False, name=name+'_groupconv{}'.format(g+1))(x_g)
            group_list.append(x_g)
        x = Concatenate(name=name+'_groupconcat')(group_list)
    
    x = BatchNormalization(name=name+'_norm2')(x)
    x = Activation('relu', name=name+'_acti2')(x)

    x = Conv2D(features, 1, strides=1, padding='valid', use_bias=False, name=name+'_conv3')(x)
    x = BatchNormalization(name=name+'_norm3')(x)
    x = Add(name=name+'_add')([x, shortcut])
    x = Activation('relu', name=name+'_acti3')(x)
    return x


##########################
# AnyNet
##########################
def AnyNet(args, name=None, **kwargs):
    img_inputs = Input(shape=(args.img_size, args.img_size, 3), name='main_input')
    cx = {"h": args.img_size, "w": args.img_size, "flops": 0, "params": 0, "acts": 0}

    ##########################
    # Stem
    ##########################
    if 'cifar' in args.dataset:
        if args.stem == 'simple':
            x = Conv2D(args.stem_out, 3, strides=1, padding='same', use_bias=False, name=name+'stem_conv')(img_inputs)
            x = BatchNormalization(name=name+'stem_norm')(x)
            x = Activation('relu', name='stem_acti')(x)
        else:
            raise ValueError()
    else:
        if args.stem == 'simple':
            x = Conv2D(args.stem_out, 3, strides=2, padding='same', use_bias=False, name=name+'stem_conv')(img_inputs)
            x = BatchNormalization(name=name+'stem_norm')(x)
            x = Activation('relu', name='stem_acti')(x)
        elif args.stem == 'resnet':
            x = Conv2D(args.stem_out, 7, strides=2, padding='same', use_bias=False, name=name+'stem_conv')(img_inputs)
            x = BatchNormalization(name=name+'stem_norm')(x)
            x = Activation('relu', name='stem_acti')(x)
            x = MaxPool2D(pool_size=3, strides=3, padding='same', name='stem_pool')(x)
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
    x = Dense(args.classes, name='logit')(x)
    img_outputs = Activation('softmax' if args.classes > 1 else 'sigmoid', name='main_output')(x)

    model = Model(img_inputs, img_outputs, name=name)
    return model