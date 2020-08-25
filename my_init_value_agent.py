import argparse

import h5py
from keras.optimizers import SGD
from keras.layers import Dense, Input, concatenate
from keras.models import Model
import dlgo.networks
from dlgo import rl
from dlgo import encoders


def main():
    pth = "//home//nail//Code_Go//checkpoints//"
    board_size =19
    network = 'large'
    hidden_size =512
    output_file = pth+'q_model'+'.h5'
    lr = 0.01

    encoder = encoders.get_encoder_by_name('simple', board_size)
    board_input = Input(shape=encoder.shape(), name='board_input')
    action_input = Input(shape=(encoder.num_points(),), name='action_input')

    processed_board = board_input
    network = getattr(dlgo.networks, network)
    for layer in network.layers(encoder.shape()):
        processed_board = layer(processed_board)

    board_plus_action = concatenate([action_input, processed_board])
    hidden_layer = Dense(hidden_size, activation='relu')(board_plus_action)
    value_output = Dense(1, activation='sigmoid')(hidden_layer)

    model = Model(inputs=[board_input, action_input], outputs=value_output)
    opt = SGD(lr=lr)
    model.compile(loss='mse', optimizer=opt)

    new_agent = rl.ValueAgent(model, encoder)
    with h5py.File(output_file, 'w') as outf:
        new_agent.serialize(outf)


if __name__ == '__main__':
    main()
