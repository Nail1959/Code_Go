# -*- coding: utf-8 -*-
import h5py

from keras.layers import Dense, Input
from keras.models import Model
import dlgo.networks
from dlgo import rl
from dlgo import encoders


def main():

    workdir = r'/media/nail/SSD_Disk/Models/'

    board_size = 19
    network = 'large'
    hidden_size = 512
    output_file = workdir+'ac_agent.h5'

    encoder = encoders.get_encoder_by_name('simple', board_size)
    board_input = Input(shape=encoder.shape(), name='board_input')

    processed_board = board_input
    network = getattr(dlgo.networks, network)
    for layer in network.layers(encoder.shape()):
        processed_board = layer(processed_board)

    policy_hidden_layer = Dense(hidden_size, activation='relu')(processed_board)
    policy_output = Dense(encoder.num_points(), activation='softmax')(policy_hidden_layer)

    value_hidden_layer = Dense(hidden_size, activation='relu')(processed_board)
    value_output = Dense(1, activation='tanh')(value_hidden_layer)

    model = Model(inputs=[board_input], outputs=[policy_output, value_output])

    new_agent = rl.ACAgent(model, encoder)
    with h5py.File(output_file, 'w') as outf:
        new_agent.serialize(outf)


if __name__ == '__main__':
    main()
