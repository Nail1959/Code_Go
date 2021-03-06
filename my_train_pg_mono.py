# -*- coding: utf-8 -*-
import h5py

from dlgo import agent
from dlgo import rl
import os, fnmatch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


    learning_agent = agent.load_policy_agent(h5py.File(learning_agent, "r"))

    i = 1
    num_files = len(experience)
    for exp_filename in experience:
        print(50*'=')
        print('Файл для обучения: %s...' % exp_filename)
        print(50 * '=')
        exp_buffer = rl.load_experience(h5py.File(exp_filename, "r"))
        learning_agent.train(exp_buffer, lr=lr, batch_size=bs)
        print('Обработано файлов: ', i, ' из ', num_files)
        i += 1

    with h5py.File(agent_out, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


if __name__ == '__main__':
    main()
