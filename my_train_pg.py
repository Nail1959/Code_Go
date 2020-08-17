# -*- coding: utf-8 -*-
import h5py

from dlgo import agent
from dlgo import rl


def main():
    pth = 'E:\\Proj_GO\\Code_Go\\checkpoints\\'
    pth_experience = 'E:\\Proj_GO\\Experience\\'
    experience = [pth_experience+'exp1_300.h5',
                  pth_experience+'exp2_01_100.h5',
                  pth_experience+'exp3_01_100.h5',
                  pth_experience+'exp4_01_100.h5',
                  pth_experience+'exp5_01_100.h5',
                  pth_experience+'exp6_01_100.h5',
                  pth_experience+'exp7_01_100.h5',
                  pth_experience+'exp8_01_100.h5',
                  pth_experience+'exp9_01_100.h5',
                  ]
    learning_agent = input('learning_agent:')
    learning_agent = pth + learning_agent+'.h5'
    print('learning_agent: ', learning_agent)
    agent_out = input('agent_out:')
    agent_out = pth + agent_out+'.h5'
    print('agent_out: ', agent_out)
    try:
        lr = float(input('lr = '))
    except:
        lr = 0.0001
    try:
        bs = int(input('bs = '))
    except:
        bs = 512


    learning_agent = agent.load_policy_agent(h5py.File(learning_agent, 'r'))
    for exp_filename in experience:
        print('Training with %s...' % exp_filename)
        exp_buffer = rl.load_experience(h5py.File(exp_filename, "r"))
        learning_agent.train(exp_buffer, lr=lr, batch_size=bs)

    with h5py.File(agent_out, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


if __name__ == '__main__':
    main()
