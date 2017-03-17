'''
This code is part of the Keras VGG-16 model
'''
from __future__ import print_function
from __future__ import absolute_import

from keras.models import Model
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import AtrousConvolution2D
from keras.utils.data_utils import get_file
from keras import backend as K

TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'


def dcn_vgg(input_tensor=None):
    input_shape = (3, None, None)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # conv_1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # conv_2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # conv_3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', border_mode='same')(x)

    # conv_4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool', border_mode='same')(x)

    # conv_5
    x = AtrousConvolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1', atrous_rate=(2, 2))(x)
    x = AtrousConvolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2', atrous_rate=(2, 2))(x)
    x = AtrousConvolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3', atrous_rate=(2, 2))(x)

    # Create model
    model = Model(img_input, x)

    # Load weights
    weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5', TH_WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')
    model.load_weights(weights_path)

    return model
