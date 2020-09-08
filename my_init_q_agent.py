# -*- coding: utf-8 -*-
from dlgo import encoders
from dlgo.goboard_fast  import GameState, Player, Point

from keras.optimizers import SGD
from keras.layers import Conv2D,Dense, Dropout, Flatten,Input
from keras.layers import ZeroPadding2D, concatenate
from keras.models import Model
import h5py
import os, fnmatch
import os.path
import numpy as np
import datetime
import tensorflow as tf

from dlgo import rl
from dlgo import kerasutil
from dlgo import scoring
from collections import namedtuple
import shutil

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

def load_agent(filename):
    with h5py.File(filename, 'r') as h5file:
        return rl.load_q_agent(h5file)

def play_games(agent1_fname, agent2_fname,
               num_games=480, board_size=19,
               gpu_frac=0.95, temperature=0.0):
    kerasutil.set_gpu_memory_target(gpu_frac)

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
            print('Agent 1 wins , time is %s' % (datetime.datetime.now()))
            wins += 1
        else:
            print('Agent 2 wins, time is %s' % (datetime.datetime.now()))
            losses += 1
        print('Agent 1 record: %d/%d' % (wins, wins + losses))
        color1 = color1.other
    return wins, losses

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

    #print_board(game.board)    Печать доски излишня.
    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )

def do_self_play(board_size, agent1_filename, agent2_filename,
                 num_games, temperature,
                 experience_filename, chunk = 100,
                 gpu_frac=0.95):
    kerasutil.set_gpu_memory_target(gpu_frac)

    agent1 = load_agent(agent1_filename)
    agent1.set_temperature(temperature)
    agent1.set_policy('eps-greedy')
    agent2 = load_agent(agent2_filename)
    agent2.set_temperature(temperature)
    agent2.set_policy('eps-greedy')

    color1 = Player.black
    times =int(num_games/chunk)

    for current_chunk in range(times):
        print('Текущая порция %d' % current_chunk)
        collector1 = rl.ExperienceCollector()
        for i in range(chunk):
            print('Симуляция игры %d/%d...' % (i + 1, chunk))
            collector1.begin_episode()
            agent1.set_collector(collector1)

            if color1 == Player.black:
                black_player, white_player = agent1, agent2
            else:
                white_player, black_player = agent1, agent2
            game_record = simulate_game(black_player, white_player, board_size)
            if game_record.winner == color1:
                print('Agent 1 wins, time is %s' % (datetime.datetime.now()))
                collector1.complete_episode(reward=1)
            else:
                print('Agent 2 wins, time is %s' % (datetime.datetime.now()))
                collector1.complete_episode(reward=-1)
            color1 = color1.other

        experience = rl.combine_experience([collector1])
        print('Saving experience buffer to %s\n' % (experience_filename+'_' + str((current_chunk+1)*chunk)+'.h5'))
        with h5py.File(experience_filename+'_' + str((current_chunk+1)*chunk)+'.h5', 'w') as experience_outf:
            experience.serialize(experience_outf)

def generator_q(experience= [], num_moves=361, batch_size=512):
    for exp_filename in experience:
        #print('Файл  с играми для обучения: %s...' % exp_filename)
        exp_buffer = rl.load_experience(h5py.File(exp_filename, "r"))

        n = exp_buffer.states.shape[0] # Количество состояний доски
        states = exp_buffer.states     # Состояния доски
        y = np.zeros((n,))             # Целевой вектор для обучения
        actions = np.zeros((n, num_moves))  # Действия к каждому состоянию доски

        for i in range(n):
            action = exp_buffer.actions[i]
            reward = exp_buffer.rewards[i]
            actions[i][action] = 1
            y[i] = reward

        while states.shape[0] >= batch_size:
            states_batch, states = states[:batch_size], states[batch_size:]
            actions_batch, actions = actions[:batch_size], actions[batch_size:]
            y_batch, y = y[:batch_size], y[batch_size:]

            yield [states_batch, actions_batch], y_batch

def get_num_samples(experience=[],num_moves=361, batch_size=512):
    num_samples = 0
    for X, y in generator_q(experience=experience,num_moves=num_moves,batch_size=batch_size):
        num_samples += y.shape[0]
    return num_samples

def main():

    board_size = 19
    hidden_size = 1024
    workdir = '//media//nail//SSD_Disk//Models//'
    output_file = workdir + 'q_agent.h5'
    lr = 0.01
    batch_size = 1024

    pth = "//media//nail//SSD_Disk//Models//"
    pth_experience = '//media//nail//SSD_Disk//Experience//'
    experience_filename = pth_experience+'exp'

    new_form_games = input('Формировать новые игровые данные с моделью?(Y/N) ').lower()

    count_exp = int(input('Сколько взять файлов для первичного обучения ? '))


    # "Заполнение" данными модели обучения из игр
    experience = []
    os.chdir(pth_experience)
    lst_files = os.listdir(pth_experience)
    pattern = input('Паттерн для выборки файлов для обучения: ')
    if len(pattern) == 0:
        pattern = "exp*.h5"
    # ==============================================================
    # Формируем список файлов с экспериментальными игровыми данными
    len_lst_files = len(lst_files)
    if count_exp<= len_lst_files:
        for entry in lst_files[:count_exp]:
            if fnmatch.fnmatch(entry, pattern):
                experience.append(entry)
    else:
        for entry in lst_files:
            if fnmatch.fnmatch(entry, pattern):
                experience.append(entry)
    # Получили список файлов игр для обучения
    # Сортировка для удобства файлов.
    if len(experience) > 0:
        experience.sort()

    else:
        print(' Нет файлов в папке для обучения!!!!')

        exit(2)

    encoder = encoders.get_encoder_by_name('simple', board_size)
    board_input = Input(shape=encoder.shape(), name='board_input')
    action_input = Input(shape=(encoder.num_points(),), name='action_input')

    # =============================================================
    # Сеть
    # =============================================================
    # conv_0a = ZeroPadding2D((3, 3))(board_input)
    # conv_0b = Conv2D(64, (13, 13), activation='relu')(conv_0a)

    conv_1a = ZeroPadding2D((3, 3))(board_input)  #(conv_0b)
    conv_1b = Conv2D(64, (9, 9), activation='relu')(conv_1a)

    conv_2a = ZeroPadding2D((2, 2))(conv_1b)
    conv_2b = Conv2D(64, (7, 7), activation='relu')(conv_2a)

    conv_3a = ZeroPadding2D((2, 2))(conv_2b)
    conv_3b = Conv2D(32, (5, 5), activation='relu')(conv_3a)

    # conv_4a = ZeroPadding2D((2, 2))(conv_3b)
    # conv_4b = Conv2D(48, (5, 5), activation='relu')(conv_4a)
    #
    # conv_5a = ZeroPadding2D((2, 2))(conv_4b)
    # conv_5b = Conv2D(48, (5, 5), activation='relu')(conv_5a)
    #
    # conv_6a = ZeroPadding2D((2, 2))(conv_5b)
    # conv_6b = Conv2D(32, (5, 5), activation='relu')(conv_6a)
    #
    # conv_7a = ZeroPadding2D((2, 2))(conv_6b)
    # conv_7b = Conv2D(32, (5, 5), activation='relu')(conv_7a)
    #
    # conv_8a = ZeroPadding2D((2, 2))(conv_7b)
    # conv_8b = Conv2D(32, (5, 5), activation='relu')(conv_8a)
    #
    # conv_9a = ZeroPadding2D((2, 2))(conv_8b)
    # conv_9b = Conv2D(32, (5, 5), activation='relu')(conv_9a)
    #
    # conv_10a = ZeroPadding2D((2, 2))(conv_9b)
    # conv_10b = Conv2D(32, (3, 3), activation='relu')(conv_10a)
    #
    # conv_11a = ZeroPadding2D((2, 2))(conv_10b)
    # conv_11b = Conv2D(32, (3, 3), activation='relu')(conv_11a)

    flat = Flatten()(conv_3b)
    #dense_1 = Dense(512)(flat)
    #drop_1 = Dropout(0.2)(dense_1)
    processed_board = Dense(512)(flat)


    board_plus_action = concatenate([action_input, processed_board])
    hidden_layer = Dense(hidden_size, activation='relu')(board_plus_action)
    value_output = Dense(1, activation='tanh')(hidden_layer)

    model = Model(inputs=[board_input, action_input], outputs=value_output)
    opt = SGD(lr=lr)
    model.compile(loss='mse', optimizer=opt)

     # Обучение модели fit_generator
    model.fit_generator(
            generator=generator_q(experience=experience, num_moves=361, batch_size=batch_size),
            steps_per_epoch=get_num_samples(experience=experience, num_moves=361, batch_size=batch_size) / batch_size,
            verbose=1,
            epochs=1,
            initial_epoch=0)
        # Прошлись по всем файлам

    new_agent = rl.QAgent(model, encoder)
    with h5py.File(output_file, 'w') as outf:
        new_agent.serialize(outf)

    if new_form_games == 'y':
        # Формируем список файлов с экспериментальными игровыми данными c новым впервые обученным агентом с двумя входами.
        experience = []
        os.chdir(pth_experience)
        lst_files = os.listdir(pth_experience)
        for entry in lst_files:
            #if fnmatch.fnmatch(entry, 'exp*'): журналы тоже в папку сохранения, чистка всего.
            if os.path.isfile(entry) == True:
                experience.append(entry)
        for filename in experience:
            shutil.move(filename, pth_experience + 'Exp_Save//' + filename)

        do_self_play(19,output_file,output_file,num_games=1000,temperature=0,
                     experience_filename=experience_filename,chunk=100)


if __name__ == '__main__':
    main()
