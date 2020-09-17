from __future__ import absolute_import

from keras.layers import BatchNormalization
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D


def layers(input_shape):
    return [
        Conv2D(192, (5, 5), input_shape=input_shape, padding='same', data_format='channels_first'),
        Activation('relu'),

        BatchNormalization(),

        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        BatchNormalization(),

        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        BatchNormalization(),

        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        BatchNormalization(),

        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        BatchNormalization(),

        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        BatchNormalization(),

        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        BatchNormalization(),

        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        BatchNormalization(),

        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        BatchNormalization(),


        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),


        Conv2D(64, (3, 3), padding='same', data_format='channels_first'),
        Activation('relu'),

        BatchNormalization(),

        Flatten(),
        Dense(512),

        Activation('relu'),
    ]

