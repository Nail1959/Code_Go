from __future__ import absolute_import

# tag::small_network[]
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D


def layers(input_shape):
    return [
        ZeroPadding2D(padding=3, input_shape=input_shape, data_format='channels_first'),
        Conv2D(64, (7, 7), data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),

        ZeroPadding2D(padding=3, input_shape=input_shape, data_format='channels_first'),
        Conv2D(64, (7, 7), data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),

        ZeroPadding2D(padding=2, input_shape=input_shape, data_format='channels_first'),
        Conv2D(64, (5, 5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D(padding=2, input_shape=input_shape, data_format='channels_first'),
        Conv2D(64, (5, 5), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D(padding=2, input_shape=input_shape, data_format='channels_first'),
        Conv2D(64, (5, 5), data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),

        ZeroPadding2D(padding=2, input_shape=input_shape, data_format='channels_first'),
        Conv2D(64, (5, 5), data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),

        ZeroPadding2D(padding=2, input_shape=input_shape, data_format='channels_first'),
        Conv2D(64, (5, 5), data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),

        ZeroPadding2D(padding=2, input_shape=input_shape, data_format='channels_first'),
        Conv2D(64, (5, 5), data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

       #BatchNormalization(),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (3, 3), data_format='channels_first'),
        Activation('relu'),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (3, 3), data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),

        Flatten(),
        Dense(1024),

        Activation('relu'),
    ]

