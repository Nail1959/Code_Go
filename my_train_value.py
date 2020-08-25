# -*- coding: utf-8 -*-
import h5py

import numpy as np

from keras.optimizers import SGD
from dlgo.encoders.simple import SimpleEncoder

from keras.layers import Dense, Input, concatenate
from keras.models import Model
import dlgo.networks
from dlgo import encoders
from dlgo import kerasutil
#from dlgo import agent
from dlgo import rl
import os, fnmatch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_q_model(pth="//home//nail//Code_Go//checkpoints//", board_size=19,
                   network='large', hidden_size=512, lr = 0.01):
    output_file = pth + 'q_model' + '.h5'

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
    return new_agent

def my_train_q(model, encoder, experience, lr=0.1, batch_size=128):
        # opt = SGD(lr=lr)
        # model.compile(loss='mse', optimizer=opt)

        n = experience.states.shape[0]
        num_moves = encoder.num_points()
        y = np.zeros((n,))
        actions = np.zeros((n, num_moves))
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            actions[i][action] = 1
            y[i] = 1 if reward > 0 else 0

        model.fit(
            [experience.states, actions], y,
            batch_size=batch_size,
            epochs=1)
        return model


def main():
    pth = '//home//nail//Code_Go//checkpoints//'
    pth_experience = '//home//nail//Experience//'
    experience = []
    os.chdir(pth_experience)
    lst_files = os.listdir(pth_experience)
    pattern = input('Паттерн для выборки файлов для обучения: ')
    if len(pattern) == 0:
        pattern = "exp*.h5"

    for entry in lst_files:
        if fnmatch.fnmatch(entry, pattern):
            experience.append(entry)

    experience.sort()
    learning_agent = input('learning_agent:')
    learning_agent = pth + learning_agent+'.h5'
    print('learning_agent: ', learning_agent)
    agent_out = input('agent_out:')
    agent_out = pth + agent_out+'.h5'
    board_size = 19
    print('agent_out: ', agent_out)
    try:
        lr = float(input('lr = '))
    except:
        lr = 0.000001
    try:
        bs = int(input('bs = '))
    except:
        bs = 1024

    # ==================================================
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.98
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    # ==================================================
    encoder = SimpleEncoder((board_size, board_size))
    try:
        h5file = h5py.File(learning_agent, "r")
        learning_agent = rl.load_q_agent(h5file)
        model_q = kerasutil.load_model_from_hdf5_group(h5file['model'])

    except:

        learning_agent = create_q_model(lr = lr)
    i = 1
    num_files = len(experience)
    for exp_filename in experience:
        print(50*'=')
        print('Файл для обучения: %s...' % exp_filename)
        print(50 * '=')
        exp_buffer = rl.load_experience(h5py.File(exp_filename, "r"))
        model_q = my_train_q(model_q, encoder, exp_buffer, lr=lr, batch_size=bs)

        print('Обработано файлов: ', i, ' из ', num_files)
        i += 1

    learning_agent = rl.QAgent(model_q,encoder)
    with h5py.File(agent_out, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


if __name__ == '__main__':
    main()
