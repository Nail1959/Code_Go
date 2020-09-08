# -*- coding: utf-8 -*-
import os,fnmatch
import numpy as np

import tensorflow as tf
import random
import h5py
from collections import namedtuple

from dlgo import kerasutil
from dlgo import scoring
from dlgo import rl
from dlgo.goboard_fast  import GameState, Player, Point
import datetime
import shutil
import time
from scipy.stats import binom_test
import os.path
import gc


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

    #print_board(game.board)    Печать доски излишня.
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

# ==================================================
def do_self_play(board_size, agent1_filename, agent2_filename,
                 num_games, temperature,
                 experience_filename, chunk = 100,
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
        print('Saving experience buffer to %s\n' % (experience_filename + str(current_chunk*chunk)+'.h5'))
        with h5py.File(experience_filename+'_' + str(current_chunk*chunk)+'.h5', 'w') as experience_outf:
            experience.serialize(experience_outf)

def eval(learning_agent, reference_agent,
             num_games = 200,
             board_size=19,
             temperature=0.0):

    game_results = play_games(learning_agent, reference_agent,
                              num_games, board_size, temperature)
    print('GameResults: ', game_results)
    total_wins, total_losses = game_results

    print('FINAL RESULTS:')
    print('Learner: %d' % total_wins)
    print('Refrnce: %d' % total_losses)

    return total_wins

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

#==========================================================================================
#   Главная управляющая программа
#==========================================================================================
def main():
    pth = "//media//nail//SSD_Disk//Models//"
    pth_experience = '//media//nail//SSD_Disk//Experience//'
    # board_size =19
    # hidden_size =512   # Можно экспериментировать. В книге было 256
    learning_agent = input('Агент для обучения "ценность действия": ')
    num_games = int(input('Количество игр для формирования учебных данных = '))
    try:
        chunk = int(input('Порция игр в одном файле для обучения = '))
    except:
        chunk = 100

    try:
        delta_games = float(input('Приращение количества игр(%) = '))/100
    except:
        delta_games = 0.4

    learning_agent = pth+learning_agent+'.h5'  # Это начальный агент либо первый либо для продолжения обучения
    output_file = pth+'new_q_agent.h5'     # Это будет уже агент обновляемый с лучшими результатами игр
    current_agent = pth + 'current_agent.h5' # Текущий обучаемый агент
    try:
        lr = float(input('Скорость обучения lr = '))
    except:
        lr = 0.01

    temp_decay = 0.98
    min_temp = 0.001
    try:
        temperature = float(input('Temperature = '))
    except:
        temperature = min_temp
    try:
        batch_size = int(input("batch_size = "))
    except:
        batch_size = 512
    try:
        epochs = int(input('Epochs = '))
    except:
        epochs = 1

    log_file = input('Журнал обработки: ')
    log_file = pth_experience + log_file+'.txt'

    logf = open(log_file, 'a')
    logf.write('----------------------\n')
    logf.write('Начало обучения агента %s в %s\n' % (
        learning_agent, datetime.datetime.now()))

    # Строится модель для обучения ценность действия
    # Два входа, один выход.
    # Компиляция модели если еще нет модели для ценности действия.
    # Иначе загружаем уже существующую модель.

    if ('q_agent' in learning_agent and os.path.isfile(learning_agent)) == False:
        exit(10)

    pattern = input('Паттерн для выборки файлов для обучения: ')
    if len(pattern) == 0:
        pattern = "exp*.h5"

    total_work = 0  # Счетчик "прогонов" обучения.
    while True:  # Можно всегда прервать обучение и потом продолжть снова.

        l_agent = load_agent(learning_agent)
        model = l_agent.model
        encoder = l_agent.encoder
        num_moves = encoder.num_points()

        logf.write('Прогон = %d\n' % total_work)
        print(50 * '=')
        experience = []
        os.chdir(pth_experience)
        lst_files = os.listdir(pth_experience)
        # ==============================================================
        # Формируем список файлов с экспериментальными игровыми данными
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

        experience.sort()

        # Обучение модели fit_generator
        model.fit_generator(
                generator=generator_q(experience=experience,num_moves=361,batch_size=batch_size),
                steps_per_epoch=get_num_samples(experience=experience,num_moves=361,batch_size=batch_size)/batch_size,
                verbose=1,
                epochs=epochs)
        # Прошлись по всем файлам

#---------------------------------------------------------------------------
        # Сохранение обученного агента нужно для оценки улучшения старого агента
        # Если агент в игре будет лучше то он сохранится в out_file и новый обучаемый агент
        # будет загружаться из обновленного с новыми игровыми данными проведенными этим
        # обновленным агентом. Если нет, старый агент без изменений, только добавить
        # дополнительное количество выполненных игр обучаемым агентом к прежним.
        new_agent = rl.QAgent(model, encoder)
        with h5py.File(current_agent, 'w') as outf:  # Сохраняем агента как текущего
            new_agent.serialize(outf)


        print('\n Оцениваем нового  агента с "старым" агентом\n')
        wins = eval(current_agent, learning_agent, num_games=200)
        print('Выиграно %d / %s игр (%.3f)' % (
            wins, str(200), float(wins) / 200))
        logf.write('Выиграно %d / %s игр (%.3f)\n' % (
            wins, str(200), float(wins) / float(200)))
        bt = binom_test(wins, 200, 0.5)*100
        print('Бином тест = ', bt , '%')
        logf.write('Бином тест = %f\n' % bt)
        if bt <= 5 and wins > 115:
            print('Обновление агента!!!!!')
            # Сохраняем обученного агента
            new_agent = rl.QAgent(model, encoder)
            with h5py.File(output_file, 'w') as outf:
                new_agent.serialize(outf)

            logf.write('Выполнено обновление агента после успешного обучения %d  время  %s\n' % (total_work,
                                                              datetime.datetime.now()))
            logf.write('Новый агент : %s\n' % output_file)

            # Очистка каталога с данными игр "старого" агента
            pth_exp_save = pth_experience+'Exp_Save'
            os.chdir(pth_exp_save)
            lst_files = os.listdir(pth_exp_save)
            for entry in lst_files:
                os.remove(entry)

            # Сохраняем игровые данные в каталоге сохранения
            experience = []
            os.chdir(pth_experience)
            lst_files = os.listdir(pth_experience)
            for entry in lst_files:
                if fnmatch.fnmatch(entry, 'exp*'):
                    experience.append(entry)
            for exp_filename in experience:
                shutil.move(exp_filename, pth_experience+'Exp_Save//'+exp_filename)

            # Формируем список файлов с экспериментальными игровыми данными
            temperature = max(min_temp, temp_decay * temperature)
            exp_filename = 'exp'+str(total_work)+'_'
            do_self_play(19, output_file, output_file, num_games=num_games,
                         temperature=temperature, experience_filename=exp_filename, chunk=chunk)
            learning_agent = output_file    # Теперь на следующей шаге обучаемым станет обновленный агент

            logf.write('Новая "температура" = %f\n' % temperature)
        else:
            # print('Агента не меняем, Игровые данные увеличивам \n')
            if num_games < 40000:
            # Добавим порцию игр
            #learning_agent оставляем без изменений. Новый обученный агент не лучше старого.
            # Добавим порцию игр для дополнительного обучения.
               add_games = int(num_games * delta_games)
               print(' Добавили дополнительное количество игр для обучения = ', add_games)
               exp_filename = 'exp' + str(total_work) + '_'
               do_self_play(19, learning_agent, learning_agent,
                         num_games=add_games,  # Добавим новые файлы к уже существующим новые файлы.
                         temperature=temperature, experience_filename=exp_filename, chunk=chunk)


        total_work += 1
        print('Количество выполненных прогонов = ', total_work)
        logf.write('Выполнен прогон %d  время at %s\n' % (total_work,
            datetime.datetime.now()))

        logf.flush()
        if total_work > 1000:
            break

if __name__ == '__main__':
    main()
