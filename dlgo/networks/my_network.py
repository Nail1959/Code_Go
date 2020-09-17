from __future__ import absolute_import

# tag::small_network[]
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D


def layers(input_shape):
    return [
        Conv2D(192, (5, 5), padding='same', data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),

        Conv2D(64, (3, 3), padding='sme', data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),

        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),


        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),


        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),


        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),

        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),

        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

       #BatchNormalization(),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),


        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),


        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        #BatchNormalization(),

        Flatten(),
        Dense(512),

        Activation('relu'),
    ]

