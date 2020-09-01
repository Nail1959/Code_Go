import argparse
import datetime
import multiprocessing

import random
import shutil
import time
import tempfile
from collections import namedtuple

import h5py
import numpy as np

from dlgo import agent
from dlgo import kerasutil
from dlgo import scoring
from dlgo import rl
from dlgo.goboard_fast import GameState, Player, Point

import os, fnmatch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}


def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))


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


def name(player):
    if player == Player.black:
        return 'B'
    return 'W'


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

  #  print_board(game.board)
    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix='dlgo-train')
    os.close(fd)
    return fname


def do_self_play(board_size, agent1_filename, agent2_filename,
                 num_games, temperature,
                 experience_filename,
                 gpu_frac):
    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    with h5py.File(agent1_filename, 'r') as agent1f:
        agent1 = agent.load_policy_agent(agent1f)
    agent1.set_temperature(temperature)
    with h5py.File(agent2_filename, 'r') as agent2f:
        agent2 = agent.load_policy_agent(agent2f)

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





def train_worker(learning_agent, output_file, experience_file,
                 lr, batch_size):
    with h5py.File(learning_agent, 'r') as learning_agentf:
        learning_agent = agent.load_policy_agent(learning_agentf)
    with h5py.File(experience_file, 'r') as expf:
        exp_buffer = rl.load_experience(expf)
    learning_agent.train(exp_buffer, lr=lr, batch_size=batch_size)

    with h5py.File(output_file, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


def train_on_experience(learning_agent, output_file, experience_file,
                        lr, batch_size):
    # Do the training in the background process. Otherwise some Keras
    # stuff gets initialized in the parent, and later that forks, and
    # that messes with the workers.
    worker = multiprocessing.Process(
        target=train_worker,
        args=[
            learning_agent,
            output_file,
            experience_file,
            lr,
            batch_size
        ]
    )
    worker.start()
    worker.join()


def play_games(args):
    agent1_fname, agent2_fname, num_games, board_size, gpu_frac = args

    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    with h5py.File(agent1_fname, 'r') as agent1f:
        agent1 = agent.load_policy_agent(agent1f)
    with h5py.File(agent2_fname, 'r') as agent2f:
        agent2 = agent.load_policy_agent(agent2f)

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


def evaluate(learning_agent, reference_agent,
             num_games, num_workers, board_size):
    games_per_worker = num_games // num_workers
    gpu_frac = 0.95 / float(num_workers)
    pool = multiprocessing.Pool(num_workers)
    worker_args = [
        (
            learning_agent, reference_agent,
            games_per_worker, board_size, gpu_frac,
        )
        for _ in range(num_workers)
    ]
    game_results = pool.map(play_games, worker_args)

    total_wins, total_losses = 0, 0
    for wins, losses in game_results:
        total_wins += wins
        total_losses += losses
    print('FINAL RESULTS:')
    print('Learner: %d' % total_wins)
    print('Refrnce: %d' % total_losses)
    pool.close()
    pool.join()
    return total_wins


def main():
    # ==================================================
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.98
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    # ==================================================

    pth = '//home//nail//Code_Go//checkpoints//'
    pth_experience = '//home//nail//Experience//'
    board_size = 19
    experience = []
    min_temp = 0.0001
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
    learning_agent = pth + learning_agent + '.h5'
    print('learning_agent: ', learning_agent)
    agent_out = input('agent_out:')
    agent_out = pth + agent_out + '.h5'
    print('agent_out: ', agent_out)
    try:
        lr = float(input('lr = '))
    except:
        lr = 0.000001

    log_file = input('Log file = ')
    try:
        temperature = float(input('Temperature = '))
    except:
        temperature = min_temp
    try:
        batch_size = int(input("batch_size = "))
    except:
        batch_size = 512

    games_per_batch = 100
    num_workers = 1


    logf = open(log_file, 'a')
    logf.write('----------------------\n')
    logf.write('Starting from %s at %s\n' % (
        learning_agent, datetime.datetime.now()))


    reference_agent = learning_agent
    experience_file = pth_experience + 'exp_temp.hdf5'
    tmp_agent = pth + 'agent_temp.hdf5'
    working_agent = pth + 'agent_cur.hdf5'
    total_games = 0
    while True:
        print('Reference: %s' % (reference_agent,))
        logf.write('Total games so far %d\n' % (total_games,))
        generate_experience(
            learning_agent, reference_agent,
            experience_file,
            num_games=games_per_batch,
            board_size=board_size,
            num_workers=num_workers,
            temperature=temperature)
        train_on_experience(
            learning_agent, tmp_agent, experience_file,
            lr=lr, batch_size=bs)
        total_games += games_per_batch
        wins = evaluate(
            learning_agent, reference_agent,
            num_games=480,
            num_workers=num_workers,
            board_size=board_size)
        print('Won %d / 480 games (%.3f)' % (
            wins, float(wins) / 480.0))
        logf.write('Won %d / 480 games (%.3f)\n' % (
            wins, float(wins) / 480.0))
        shutil.copy(tmp_agent, working_agent)
        learning_agent = working_agent
        if wins >= 262:
            next_filename = os.path.join(
                pth,
                'agent_%08d.hdf5' % (total_games,))
            shutil.move(tmp_agent, next_filename)
            reference_agent = next_filename
            logf.write('New reference is %s\n' % next_filename)
        else:
            print('Keep learning\n')
        logf.flush()


if __name__ == '__main__':
    main()
