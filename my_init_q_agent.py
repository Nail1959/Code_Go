import argparse

import h5py

from keras.layers import Dense, Input, concatenate
from keras.models import Model
import dlgo.networks
from dlgo import rl
from dlgo import encoders

from keras.optimizers import SGD
from keras.layers import Conv2D,Dense, Flatten,Input
from keras.layers import ZeroPadding2D, concatenate
from keras.models import Model
import h5py

from dlgo import rl



def main():

    board_size = 19
    hidden_size = 512
    workdir = '//home//nail//Code_Go//checkpoints//'
    output_file = workdir + 'q_agent.h5'


    encoder = encoders.get_encoder_by_name('simple', board_size)
    board_input = Input(shape=encoder.shape(), name='board_input')
    action_input = Input(shape=(encoder.num_points(),), name='action_input')

    # =============================================================
    # Сеть
    # =============================================================
    conv_1a = ZeroPadding2D((3, 3))(board_input)
    conv_1b = Conv2D(64, (7, 7), activation='relu')(conv_1a)

    conv_2a = ZeroPadding2D((2, 2))(conv_1b)
    conv_2b = Conv2D(64, (5, 5), activation='relu')(conv_2a)

    conv_3a = ZeroPadding2D((2, 2))(conv_2b)
    conv_3b = Conv2D(64, (5, 5), activation='relu')(conv_3a)

    conv_4a = ZeroPadding2D((2, 2))(conv_3b)
    conv_4b = Conv2D(48, (5, 5), activation='relu')(conv_4a)

    conv_5a = ZeroPadding2D((2, 2))(conv_4b)
    conv_5b = Conv2D(48, (5, 5), activation='relu')(conv_5a)

    conv_6a = ZeroPadding2D((2, 2))(conv_5b)
    conv_6b = Conv2D(32, (5, 5), activation='relu')(conv_6a)

    conv_7a = ZeroPadding2D((2, 2))(conv_6b)
    conv_7b = Conv2D(32, (5, 5), activation='relu')(conv_7a)

    flat = Flatten()(conv_7b)
    processed_board = Dense(512)(flat)


    board_plus_action = concatenate([action_input, processed_board])
    hidden_layer = Dense(hidden_size, activation='relu')(board_plus_action)
    value_output = Dense(1, activation='tanh')(hidden_layer)

    model = Model(inputs=[board_input, action_input], outputs=value_output)

    new_agent = rl.QAgent(model, encoder)
    with h5py.File(output_file, 'w') as outf:
        new_agent.serialize(outf)


if __name__ == '__main__':
    main()
