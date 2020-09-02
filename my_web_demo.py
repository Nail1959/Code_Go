import h5py

from dlgo import agent
from dlgo import httpfrontend
from dlgo import mcts
from dlgo import rl
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#==================================================
#
#
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
#==================================================

def main():
    workdir = '//home/nail//Code_Go//checkpoints//'
    os.chdir(workdir)
    bind_address = '127.0.0.1'
    port = 5000
    predict_agent,pg_agent,q_agent,ac_agent = '','','',''
    agent_type = input('Агент(pg/predict/q/ac = ').lower()
    if agent_type == 'pg':
        pg_agent = input('Введите имя файла для игры с ботом политика градиентов =')
        pg_agent = workdir+pg_agent+'.h5'
    if agent_type == 'predict':
        predict_agent = input('Введите имя файла для игры с ботом предсказания хода =')
        predict_agent = workdir+predict_agent+'.h5'
    if agent_type == 'q':
        q_agent = input('Введите имя файла для игры с ботом ценность действия =')
        q_agent = workdir+q_agent+'.h5'
    if agent_type == 'ac':
        ac_agent = input('Введите имя файла для игры с ботом актор-критик =')
        ac_agent = workdir+ac_agent+'.h5'


    bots = {'mcts': mcts.MCTSAgent(800, temperature=0.7)}
    if agent_type == 'pg':
        bots['pg'] = agent.load_policy_agent(h5py.File(pg_agent,'r'))
    if agent_type == 'predict':
        bots['predict'] = agent.load_prediction_agent(h5py.File(predict_agent,'r'))
    if agent_type == 'q':
        q_bot = rl.load_q_agent(h5py.File(q_agent,'r'))
        q_bot.set_temperature(0.01)
        bots['q'] = q_bot
    if agent_type == 'ac':
        ac_bot = rl.load_ac_agent(h5py.File(ac_agent,'r'))
        ac_bot.set_temperature(0.05)
        bots['ac'] = ac_bot

    web_app = httpfrontend.get_web_app(bots)
    web_app.run(host=bind_address, port=port, threaded=False)


if __name__ == '__main__':
    main()
