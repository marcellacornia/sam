'''
This code is part of the Keras ResNet-50 model
'''
from __future__ import print_function
from __future__ import absolute_import

from keras.layers import merge, Input, Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import AtrousConvolution2D
from keras.layers import BatchNormalization
from keras. models import Model
from keras import backend as K
from keras.utils.data_utils import get_file

TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block_atrous(input_tensor, kernel_size, filters, stage, block, atrous_rate=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = AtrousConvolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                            atrous_rate=atrous_rate, name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def identity_block_atrous(input_tensor, kernel_size, filters, stage, block, atrous_rate=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = AtrousConvolution2D(nb_filter2, kernel_size, kernel_size, atrous_rate=atrous_rate,
                            border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def dcn_resnet(input_tensor=None):
    input_shape = (3, None, None)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor

    bn_axis = 1

    # conv_1
    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same')(x)

    # conv_2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # conv_3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(2, 2))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # conv_4
    x = conv_block_atrous(x, 3, [256, 256, 1024], stage=4, block='a', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='b', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='c', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='d', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='e', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='f', atrous_rate=(2, 2))

    # conv_5
    x = conv_block_atrous(x, 3, [512, 512, 2048], stage=5, block='a', atrous_rate=(4, 4))
    x = identity_block_atrous(x, 3, [512, 512, 2048], stage=5, block='b', atrous_rate=(4, 4))
    x = identity_block_atrous(x, 3, [512, 512, 2048], stage=5, block='c', atrous_rate=(4, 4))

    # Create model
    model = Model(img_input, x)

    # Load weights
    weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels_notop.h5', TH_WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models', md5_hash='f64f049c92468c9affcd44b0976cdafe')
    model.load_weights(weights_path)

    return model