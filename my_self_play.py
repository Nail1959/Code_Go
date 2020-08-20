#import argparse
import datetime
from collections import namedtuple

import h5py

from dlgo import agent
from dlgo import scoring
from dlgo import rl
from dlgo.goboard_fast import GameState, Player, Point
import os


BOARD_SIZE = 19
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
    for row in range(BOARD_SIZE, 0, -1):
        line = []
        for col in range(1, BOARD_SIZE + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%2d %s' % (row, ''.join(line)))
    print('   ' + COLS[:BOARD_SIZE])


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def name(player):
    if player == Player.black:
        return 'B'
    return 'W'


def simulate_game(black_player, white_player):
    moves = []
    game = GameState.new_game(BOARD_SIZE)
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


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--learning-agent', required=True)
    # parser.add_argument('--num-games', '-n', type=int, default=10)
    # parser.add_argument('--game-log-out', required=True)
    # parser.add_argument('--experience-out', required=True)
    # parser.add_argument('--temperature', type=float, default=0.0)

    learning_agent = input("Бот: ")
    temperature = float(input('Температура = '))
    game_log = input('game_log: ')
    experience_out = input('experience_out: ')
    num_games = int(input('Количество игр = '))
    try:
        chunk_size = int(input('Количество игр в "порции" ='))
    except:
        chunk_size = 100

    pth = "//home//nail//Experience//"
    learning_agent = '//home//nail//Code_Go//checkpoints//'+learning_agent+'.h5'
    game_log = pth+game_log + '_' + str(num_games)
    experience_out = pth+experience_out+'_' + str(num_games)+'_' #+'.h5'

    #args = parser.parse_args()
    # ==================================================
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    # ==================================================

    agent1 = agent.load_policy_agent(h5py.File(learning_agent, "r"))
    agent2 = agent.load_policy_agent(h5py.File(learning_agent, "r"))
    agent1.set_temperature(temperature)
    agent2.set_temperature(temperature)

    k = 0
    j = 0
    for i in range(num_games+1):
        if j == 0:
            game_log_out = game_log+'_'+str((k+1)*chunk_size)+".txt"
            logf = open(game_log_out, 'a')
            logf.write('Начало игр в  %s\n' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),))
            logf.write(str((k+1)*chunk_size) +' из количества игр: ' + str(num_games) + '\n')
            print('Моделируемая игра %d/%d...' % (i + 1, num_games))
            collector1 = rl.ExperienceCollector()
            collector2 = rl.ExperienceCollector()

            color1 = Player.black
        j += 1
        collector1.begin_episode()
        agent1.set_collector(collector1)
        collector2.begin_episode()
        agent2.set_collector(collector2)

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent2, agent1
        game_record = simulate_game(black_player, white_player)
        print(" № игры : ", i+1)
        if game_record.winner == color1:
            print('Агент 1 выигрывает.')
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            print('Агент 2 выигрывает.')
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=-1)
        color1 = color1.other

        if i >= chunk_size and i % chunk_size ==  0:

            experience = rl.combine_experience([collector1, collector2])
            experience_out_file = experience_out+str((k+1)*chunk_size)+".h5"
            logf.write('Сохранение буфера в файл %s\n' % experience_out_file)
            logf.write('Завершение игр %s\n' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),))
            logf.close()
            with h5py.File(experience_out_file, 'w') as experience_outf:
                experience.serialize(experience_outf)
            print('Записано игр: ', (k+1)*chunk_size, ' из ', num_games, ' игр.')
            k += 1
            j = 0


if __name__ == '__main__':
    main()
