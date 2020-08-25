# -*- coding: utf-8 -*-
from keras.optimizers import SGD
from keras.layers import Dense, Input, concatenate
from keras.models import Model
import dlgo.networks
from dlgo import encoders
import os,fnmatch
import numpy as np
#from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import random
import time
import h5py
from collections import namedtuple

from dlgo import kerasutil
from dlgo import scoring
from dlgo import rl
from dlgo.goboard_fast  import GameState, Player, Point
import datetime


def load_agent(filename):
    with h5py.File(filename, 'r') as h5file:
        return rl.load_q_agent(h5file)


COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ==================================================
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
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

# ==================================================
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

def eval(learning_agent, reference_agent,
             num_games = 480,
             board_size=19,
             temperature=0.0):

    game_results = play_games(learning_agent, reference_agent,
                              num_games, board_size, temperature)

    total_wins, total_losses = 0, 0
    for wins, losses in game_results:
        total_wins += wins
        total_losses += losses
    print('FINAL RESULTS:')
    print('Learner: %d' % total_wins)
    print('Refrnce: %d' % total_losses)

    return total_wins


#==========================================================================================
#   Главная управляющая программа
#==========================================================================================
def main():
    pth = "//home//nail//Code_Go//checkpoints//"
    pth_experience = '//home//nail//Experience//'
    board_size =19
    network = 'large'
    hidden_size =512
    learning_agent = input('Агент для обучения "ценность действия": ')
    learning_agent = pth+learning_agent+'.h5'  # Это агент либо от политики градиентов(глава 10),либо из главы 7"
    output_file = pth+'value_model'+'.h5'      # Это будет уже агент с двумя входами для ценности действия
    lr = 0.01
    temp_decay = 0.98
    min_temp = 0.01
    try:
        temperature = float(input('Temperature = '))
    except:
        temperature = min_temp
    try:
        batch_size = int(input("batch_size = "))
    except:
        batch_size = 512

    log_file = input('Журнал обработки: ')
    log_file = pth_experience + log_file+'.txt'

    logf = open(log_file, 'a')
    logf.write('----------------------\n')
    logf.write('Starting from %s at %s\n' % (
        learning_agent, datetime.datetime.now()))

    # Строится модель для обучения ценность действия
    # Два входа, один выход.
    # Компиляция модели если еще нет модели для ценности действия.
    # Иначе загружаем уже существующую модель.
    try:
        q_agent = load_agent(output_file)  # Модель уже есть, надо продолжить обучение
        model = q_agent.model
        encoder = q_agent.encoder
        temperature = q_agent.temperature
    except:
        # Еще только надо создать модель для обучения
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

# "Заполнение" данными модели обучения из игр



    experience = []
    os.chdir(pth_experience)
    lst_files = os.listdir(pth_experience)
    pattern = input('Паттерн для выборки файлов для обучения: ')
    if len(pattern) == 0:
        pattern = "exp*.h5"
    #==============================================================
    # Формируем список файлов с экспериментальными игровыми данными
    for entry in lst_files:
        if fnmatch.fnmatch(entry, pattern):
            experience.append(entry)
    # Получили список файлов игр для обучения
    exp_filename = ''
    if len(experience) > 0:
        experience.sort()
        exp_filename = experience[0]  # Нужен только один файл
    else:
        print(' Нет файлов в папке для обучения!!!!')
        exit(2)

    #==============================================================
    # callback_list = [ModelCheckpoint(pth, monitor='val_accuracy',
    #                                  save_best_only=True)]

    total_work= 0   # Счетчик "прогонов" обучения.
    while True:  # Можно всегда прервать обучение и потом продолжть снова.
        logf.write('Прогон = %d\n' % total_work)
        print(50 * '=')
        print('Файл для обучения: %s...' % exp_filename)
        print(50 * '=')
        exp_buffer = rl.load_experience(h5py.File(exp_filename, "r"))
        # Заполняем данными для обучения скомпилированную модель.
        n = exp_buffer.states.shape[0]
        num_moves = encoder.num_points()
        y = np.zeros((n,))
        actions = np.zeros((n, num_moves))
        for i in range(n):
            action = exp_buffer.actions[i]
            reward = exp_buffer.rewards[i]
            actions[i][action] = 1
            y[i] = 1 if reward > 0 else 0
        # Обучение модели
        model.fit(
            [exp_buffer.states, actions], y,
            batch_size=batch_size, #verbose=1,  #callbacks=callback_list, Эпоха = 1
            epochs=1)

        # Сохраняем обученного агента
        new_agent = rl.QAgent(model, encoder)
        with h5py.File(output_file, 'w') as outf:
            new_agent.serialize(outf)
        if total_work == 0:  # Это первый агент с двумя входными данными.
            print('Обновление агента!!!!!')
            learning_agent = output_file
            os.chdir(pth_experience)

            lst_files = os.listdir(pth_experience)
            for entry in lst_files:
                if fnmatch.fnmatch(entry, "exp*"):
                    os.remove('//home//nail//Experience//'+entry)  # Очистка каталога с данными игр "старого" агента
            # Формируем новые игровые данные с новым агентом.
            do_self_play(19, output_file, output_file, num_games=200,
                         temperature=temperature, experience_filename=exp_filename)
            continue    # Сравнивать пока не с чем.

        # Сравниваем результат игры нового агента с "старым" агентом.
        wins = eval(output_file, learning_agent)
        print('Выиграно %d / 480 games (%.3f)' % (
            wins, float(wins) / 480.0))
        logf.write('Won %d / 480 games (%.3f)\n' % (
            wins, float(wins) / 480.0))
        if wins >= 262:
            print('Обновление агента!!!!!')
            learning_agent =  output_file
            os.remove('//home//nail//Experience//*')  # Очистка каталога с данными игр "старого" агента
            # Формируем новые игровые данные с новым агентом.
            do_self_play(19,output_file, output_file, num_games=200, experience_filename=exp_filename)
            temperature = max(min_temp, temp_decay * temperature)
            logf.write('New temperature is %f\n' % temperature)
        else:
            print('Агента не меняем \n')
        total_work += 1
        print('Количество выполненных прогонов = ', total_work)
        logf.flush()

if __name__ == '__main__':
    main()
