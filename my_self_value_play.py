import datetime
import os
import random
import shutil
import time
import tempfile
from collections import namedtuple

import h5py
import numpy as np

from dlgo import kerasutil
from dlgo import scoring
from dlgo import rl
from dlgo.goboard_fast import GameState, Player, Point


def load_agent(filename):
    with h5py.File(filename, 'r') as h5file:
        return rl.load_value_agent(h5file)


COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}

def do_self_play(board_size, agent1_filename, agent2_filename,
                 num_games, temperature,
                 experience_filename,
                 gpu_frac=0.95):
    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_filename)
    agent1.set_temperature(temperature)
    agent1.set_policy('eps-greedy')
    agent2 = load_agent(agent2_filename)
    agent2.set_temperature(temperature)
    agent2.set_policy('eps-greedy')

    collector1 = rl.ExperienceCollector()

    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        collector1.begin_episode()
        agent1.set_collector(collector1)

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == color1:
            print('Agent 1 wins.')
            collector1.complete_episode(reward=1)
        else:
            print('Agent 2 wins.')
            collector1.complete_episode(reward=-1)
        color1 = color1.other

    experience = rl.combine_experience([collector1])
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)

def name(player):
    if player == Player.black:
        return 'B'
    return 'W'

def print_board(board):
    for row in range(board.num_rows, 0, -1):
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%2d %s' % (row, ''.join(line)))
    print('   ' + COLS[:board.num_cols])


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def simulate_game(black_player, white_player, board_size):
    moves = []
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    print_board(game.board)
    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )

def play_games(agent1_fname, agent2_fname,
               num_games=480, board_size=19,
               gpu_frac=0.95, temperature=0.0):


    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agent1 = load_agent(agent1_fname)
    agent1.set_temperature(temperature)
    agent1.set_policy('eps-greedy')
    agent2 = load_agent(agent2_fname)
    agent2.set_temperature(temperature)
    agent2.set_policy('eps-greedy')

    wins, losses = 0, 0
    color1 = Player.black
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent1, agent2
        game_record = simulate_game(black_player, white_player, board_size)
        if game_record.winner == color1:
            print('Agent 1 wins')
            wins += 1
        else:
            print('Agent 2 wins')
            losses += 1
        print('Agent 1 record: %d/%d' % (wins, wins + losses))
        color1 = color1.other
    return wins, losses

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    # ==================================================
    work_dir = '//home//nail//Code_Go//checkpoints//'
    exp_dir = "//home//nail//Experience//"
    agent = input('Агент: ')
    agent = work_dir + agent + '.h5'
    exp_fname = input('Результат игры(паттерн): ')
    exp_fname = exp_dir+exp_fname
    num_expf = int(input('Количество файлов игр в ='))
    num_games = int(input('Количество игр в пакете ='))
    temperature = float(input('"Температура" = '))
    board_size = 19
    for i in range(num_expf):
      exp_fname = exp_fname+'_'+str(i+1)+'.h5'
      do_self_play(board_size,agent, agent,num_games,temperature,exp_fname)