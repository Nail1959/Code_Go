# -*- coding: utf-8 -*-
from keras.optimizers import SGD
from keras.layers import Conv2D,Dense, Flatten,Input
from keras.layers import ZeroPadding2D, concatenate
from keras.models import Model
from dlgo.networks import small
import os,fnmatch
import numpy as np
from dlgo.encoders.simple import SimpleEncoder
#from keras.callbacks import ModelCheckpoint
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

    collector1 = rl.ExperienceCollector()

    color1 = Player.black
    times =int(num_games/chunk)

    for current_chunk in range(times):
        print('Текущая порция %d' % current_chunk)
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
        print('Saving experience buffer to %s\n' % experience_filename+'_' + str(current_chunk)+'.h5')
        with h5py.File(experience_filename+'_' + str(current_chunk)+'.h5', 'w') as experience_outf:
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


#==========================================================================================
#   Главная управляющая программа
#==========================================================================================
def main():
    pth = "//home//nail//Code_Go//checkpoints//"
    pth_experience = '//home//nail//Experience//'
    board_size =19
    network = 'small'
    hidden_size =512   # Можно экспериментировать. В книге было 256
    learning_agent = input('Агент для обучения "ценность действия": ')
    num_games = int(input('Количество игр для формирования учебных данных = '))
    try:
        chunk = int(input('Порция игр в одном файле для обучения'))
    except:
        chunk = 100

    delta_games = int(input('Приращение количества игр = '))
    learning_agent = pth+learning_agent+'.h5'  # Это агент либо от политики градиентов(глава 10),либо из главы 7"
    output_file = pth+'new_q_agent.h5'     # Это будет уже агент с двумя входами для ценности действия
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

    if 'q_agent' in learning_agent and os.path.isfile(learning_agent):
        New_QAgent = False # Модель уже есть, надо продолжить обучение
        encoder = '' # Чтобы не было предупреждений о возможной ошибке в коде ниже.
        model = ''
    else:
        # Еще только надо создать модель для обучения
        # Нет модели с двумя входами.
        New_QAgent = True

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
    # Сортировка для удобства файлов.
    if len(experience) > 0:
        experience.sort()

    else:
        print(' Нет файлов в папке для обучения!!!!')

        exit(2)

    #=============================================================
    # callback_list = [ModelCheckpoint(pth, monitor='val_accuracy',
    #                                  save_best_only=True)]

    total_work = 0  # Счетчик "прогонов" обучения.

    while True:  # Можно всегда прервать обучение и потом продолжть снова.

        encoder = SimpleEncoder((board_size,board_size))
        board_input = Input(shape=encoder.shape(), name='board_input')
        action_input = Input(shape=(encoder.num_points(),), name='action_input')

        processed_board = board_input
         
        #=============================================================
        # Сеть
        #=============================================================
        conv_1a = ZeroPadding2D((3, 3))(board_input)
        conv_1b = Conv2D(64, (7, 7),activation='relu')(conv_1a)

        conv_2a = ZeroPadding2D((2, 2))(conv_1b)
        conv_2b = Conv2D(64, (5, 5),activation='relu')(conv_2a)

        conv_3a = ZeroPadding2D((2, 2))(conv_2b)
        conv_3b = Conv2D(64, (5, 5),activation='relu')(conv_3a)

        conv_4a = ZeroPadding2D((2, 2))(conv_3b)
        conv_4b = Conv2D(48, (5, 5), activation='relu')(conv_4a)

        conv_5a = ZeroPadding2D((2, 2))(conv_4b)
        conv_5b = Conv2D(48, (5, 5), activation='relu')(conv_5a)


        conv_6a = ZeroPadding2D((2, 2))(conv_5b)
        conv_6b = Conv2D(32, (5, 5),activation='relu')(conv_6a)

        conv_7a = ZeroPadding2D((2, 2))(conv_6b)
        conv_7b = Conv2D(32, (5, 5), activation='relu')(conv_7a)

        flat = Flatten()(conv_7b)
        processed_board = Dense(512)(flat)

        #============================================================

        board_plus_action = concatenate([action_input, processed_board])
        hidden_layer = Dense(hidden_size, activation='relu')(board_plus_action)
        value_output = Dense(1, activation='tanh')(hidden_layer)  # В книге tanh тангенс гиперболический

        model = Model(inputs=[board_input, action_input], outputs=value_output)
        opt = SGD(lr=lr)
        model.compile(loss='mse', optimizer=opt)

        logf.write('Прогон = %d\n' % total_work)
        print(50 * '=')
        for exp_filename in  experience:
            print('Файл  с играми для обучения: %s...' % exp_filename)
            exp_buffer = rl.load_experience(h5py.File(exp_filename, "r"))

            # Заполняем данными для обучения из считанного буфера с играми  скомпилированную модель.
            n = exp_buffer.states.shape[0]
            num_moves = encoder.num_points()
            y = np.zeros((n,))
            actions = np.zeros((n, num_moves))
            for i in range(n):
                action = exp_buffer.actions[i]
                reward = exp_buffer.rewards[i]
                actions[i][action] = 1
                y[i] = reward
            # Обучение модели
            model.fit(
                [exp_buffer.states, actions], y,
                batch_size=batch_size,
                epochs=epochs)
        # Прошлись по всем файлам

        if New_QAgent == True:  # Не было еще агента с двумя входами.
          print('Обновление агента!!!!! Это первый обновленный агент.')
          logf.write('Первое начальное обновление обученного агента\n')
        # Сохраняем обученного агента
          total_work += 1
          New_QAgent = False  # Теперь есть сохраненная модельс двумя входами.
          learning_agent = current_agent # Обучать будем нового созданного с двумя входами.
          new_agent = rl.QAgent(model, encoder)
          with h5py.File(current_agent, 'w') as outf:  # Сохраняем агента как текущего
              new_agent.serialize(outf)
          continue    # Сравнивать пока не с чем. Старые игровые данные оставляем

        new_agent = rl.QAgent(model, encoder)
        with h5py.File(current_agent, 'w') as outf:  # Сохраняем агента как текущего
            new_agent.serialize(outf)

        # Сравниваем результат игры нового текущего агента с "старым" агентом.
        wins = eval(current_agent, learning_agent, num_games=200)
        print('Выиграно %d / %s игр (%.3f)' % (
            wins, str(num_games), float(wins) / float(num_games)))
        logf.write('Выиграно %d / %s игр (%.3f)\n' % (
            wins, str(num_games), float(wins) / float(num_games)))
        bt = binom_test(wins, num_games, 0.5)*100
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

            os.remove('//home//nail//Experience//*')  # Очистка каталога с данными игр "старого" агента

            experience = []
            os.chdir(pth_experience)
            lst_files = os.listdir(pth_experience)

            # Формируем список файлов с экспериментальными игровыми данными
            for entry in lst_files:
                if fnmatch.fnmatch(entry, 'exp*'):
                    experience.append(entry)
            for exp_filename in experience:
                shutil.move(exp_filename, pth_experience+'Exp_Save//'+exp_filename)

            temperature = max(min_temp, temp_decay * temperature)
            exp_filename = 'exp'+str(total_work)+'_'
            do_self_play(19,output_file, output_file, num_games=num_games,
                         temperature=temperature, experience_filename=exp_filename, chunk=100)
            learning_agent = output_file

            logf.write('Новая "температура" = %f\n' % temperature)
        else:
            # print('Агента не меняем, Игровые данные увеличивам, параметр сходимости уменьшаем. \n')
            if num_games < 10000:
            # Добавим порцию игр
               exp_filename = 'exp' + str(total_work) + '_'
               do_self_play(19, learning_agent, learning_agent, num_games=delta_games,
                         temperature=temperature, experience_filename=exp_filename, chunk=100)
            lr = lr * 0.1


        total_work += 1
        print('Количество выполненных прогонов = ', total_work)
        logf.write('Выполнен прогон %d  время at %s\n' % (total_work,
            datetime.datetime.now()))

        logf.flush()
        if total_work > 1000:
            break

if __name__ == '__main__':
    main()
