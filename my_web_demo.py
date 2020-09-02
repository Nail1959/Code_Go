import argparse

import h5py

from dlgo import agent
from dlgo import httpfrontend
from dlgo import mcts
from dlgo import rl


def main():
    bind_address = '127.0.0.1'
    port = 5000
    agent = input('Агент(pg/predict/q/ac = ').lower()
    if agent == 'pg':
        pg_agent = input('Введите имя файла для игры с ботом политика градиентов =')
    if agent == 'predict':
        predict_agent = input('Введите имя файла для игры с ботом предсказания хода =')
    if agent == 'q':
        pg_agent = input('Введите имя файла для игры с ботом ценность действия =')
    if agent == 'ac':
        ac_agent = input('Введите имя файла для игры с ботом актор-критик =')



    args = parser.parse_args()

    bots = {'mcts': mcts.MCTSAgent(800, temperature=0.7)}
    if agent == 'pg':
        bots['pg'] = agent.load_policy_agent(h5py.File(pg_agent))
    if agent == 'predict':
        bots['predict'] = agent.load_prediction_agent(
            h5py.File(args.predict_agent))
    if args.q_agent:
        q_bot = rl.load_q_agent(h5py.File(args.q_agent))
        q_bot.set_temperature(0.01)
        bots['q'] = q_bot
    if args.ac_agent:
        ac_bot = rl.load_ac_agent(h5py.File(args.ac_agent))
        ac_bot.set_temperature(0.05)
        bots['ac'] = ac_bot

    web_app = httpfrontend.get_web_app(bots)
    web_app.run(host=bind_address, port=port, threaded=False)


if __name__ == '__main__':
    main()
