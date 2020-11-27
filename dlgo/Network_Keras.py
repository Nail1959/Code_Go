#-*- coding:UTF-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

from dlgo.data.parallel_processor import GoDataProcessor
#from dlgo.encoders.simple import SimpleEncoder
from dlgo.encoders.alphago import AlphaGoEncoder
#from dlgo.encoders.sevenplane import SevenPlaneEncoder
#from dlgo.encoders.my_fiveplane_s import MyFivePlaneEncoder_S
#from dlgo.encoders.betago import BetaGoEncoder
#from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.networks import large #my_network
from dlgo.networks import alphago #my_network

from keras.models import Sequential
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau,LearningRateScheduler
from keras.callbacks import TensorBoard
from keras.models import load_model
from dlgo.agent.predict import DeepLearningAgent
import os
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import math
import glob
import fnmatch
#import cProfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def bot_save(model, encoder, where_save_bot):
    # Сохранение для бота чтобы играть в браузере с ботом play_predict_19.html
    deep_learning_bot = DeepLearningAgent(model, encoder)
    model_file = h5py.File(where_save_bot, "w")
    deep_learning_bot.serialize(model_file)

def step_decay (epoch): # Параметр затухания для оптимизатора SGD.
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
   return lrate


def my_first_network(cont_train=True, num_games=100, num_samples=None, num_samples_test=None,
                     epochs=10, batch_size=128, percent_validation = 10,
                     optimizer='adadelta', learning_rate = 0.1, patience=5,
                     where_save_model = '../checkpoints/small_model_epoch_{epoch:3d}_{val_loss:.3f}_{val_accuracy:.3f}.h5',
                     where_save_bot='../checkpoints/small_deep_bot.h5',pr_kgs='n', seed =1337, name_model='my_small',
                     num_layers = 12, num_filters=10,
                     first_kernel_size=5,
                     other_kernel_size=3
                     ):
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols

    model = Sequential()


    encoder = AlphaGoEncoder((go_board_rows,go_board_cols), use_player_plane=True)

    processor = GoDataProcessor(encoder=encoder.name(), data_directory='data')

    if pr_kgs == 'y':  # Only forming train and test data   into data directory

        generator = processor.load_go_data('train', num_games, use_generator=True, seed = seed)
        test_generator = processor.load_go_data('test', num_games, use_generator=True,seed = seed)
        return

    else:
        generator = processor.load_go_data('train', num_games, use_generator=True, seed=0)
        test_generator = processor.load_go_data('test', num_games, use_generator=True,seed=0)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    if network_alphago != 'y':
        network_layers = large.layers(input_shape)


    train_log = 'training_'+name_model+'_'+str(num_games)+'_epochs_'+str(epochs)+'_'+optimizer+'.csv'
    csv_logger = CSVLogger(train_log, append=True, separator=';')
    lrate = LearningRateScheduler(step_decay)

    if patience > 2:
        r_patience = patience - 1
    else:
        r_patience = patience
    Reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=r_patience,
                               verbose=1, mode='auto', min_delta=0.00001, cooldown=0, min_lr=0)
    # tensor_board_log_dir = '/home/nail/CODE_GO/checkpoints/my_log_dir'
    # tensorboard = TensorBoard(log_dir=tensor_board_log_dir, histogram_freq=1, embeddings_freq=1, write_graph=True)

    if optimizer == 'adagrad':
        callback_list = [ModelCheckpoint(where_save_model, monitor ='val_accuracy',
                                save_best_only=True),
                         EarlyStopping(monitor='val_accuracy', mode='auto',verbose=verb, patience=patience,
                                       min_delta=0,restore_best_weights=True),
                         csv_logger,
                         Reduce
                         ]
    elif optimizer == 'SGD':
        callback_list = [ModelCheckpoint(where_save_model, monitor='val_accuracy',
                                save_best_only=True),
                         EarlyStopping(monitor='val_accuracy', mode='auto', verbose=verb, patience=patience,
                                       min_delta=0,restore_best_weights=True),
                         csv_logger,
                         Reduce,
                         lrate
                         ]
    else:
        callback_list = [ModelCheckpoint(where_save_model, monitor='val_accuracy',
                                         save_best_only=True),
                         EarlyStopping(monitor='val_accuracy', mode='auto', verbose=verb, patience=patience,
                                       min_delta=0, restore_best_weights=True),
                         csv_logger,
                         Reduce,
                         ]

    if cont_train is False: # Обучение с самого начала с случайных весов
        if network_alphago != 'y':  # Not AlphaGo network
            for layer in network_layers:
                model.add(layer)

            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                          metrics=['accuracy'])
            model.summary()
        else:  # AlphaGo network
            model = alphago.alphago_model(input_shape=input_shape,is_policy_net=True, num_filters=num_filters)
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                          metrics=['accuracy'])
            model.summary()
            model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                          metrics=['accuracy'])
            model.summary()

        # if num_samples == None:
        history = model.fit_generator(
            generator=generator.generate(batch_size, num_classes),
            epochs=epochs,
            steps_per_epoch=generator.get_num_samples() / batch_size,
            validation_data=test_generator.generate(
                batch_size, num_classes),
            validation_steps=test_generator.get_num_samples() / batch_size,
            verbose=verb,
            callbacks=callback_list
            )
        # else:
        #     step_per_ep_train = int(num_samples / batch_size)
        #     valid_steps = int(percent_validation/100 * num_samples_test / batch_size) # 10% от всех тестовых данных
        #     history = model.fit_generator(
        #         generator=generator.generate(batch_size, num_classes),
        #         epochs=epochs,
        #         steps_per_epoch=step_per_ep_train,
        #         validation_data=test_generator.generate(
        #             batch_size, num_classes),
        #         validation_steps=valid_steps,
        #         verbose=verb,
        #         callbacks=callback_list
        #     )

    if cont_train is True: # Обучение используя уже предобученную модель, продолжение обучения.

        model = load_model('../checkpoints/model_continue.h5')
        if optimizer == 'SGD':
            opt = SGD(learning_rate=learning_rate)
        if optimizer == 'adagrad':
            opt = Adagrad(learning_rate=learning_rate )
        if optimizer == 'adadelta':
            opt = Adadelta(learning_rate=learning_rate)

        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])
        # if num_samples == None:
        history = model.fit_generator(
            generator=generator.generate(batch_size, num_classes),
            epochs=epochs,
            steps_per_epoch=generator.get_num_samples() / batch_size,
            validation_data=test_generator.generate(
                batch_size, num_classes),
            validation_steps=test_generator.get_num_samples() / batch_size,
            verbose=verb,
            callbacks=callback_list
            )
        # else:
        #     step_per_ep_train = int(num_samples / batch_size)
        #     valid_steps = int(percent_validation/100 * num_samples_test / batch_size) # 10% от всех тестовых данных
        #     history = model.fit_generator(
        #         generator=generator.generate(batch_size, num_classes),
        #         epochs=epochs,
        #         steps_per_epoch=step_per_ep_train,
        #         validation_data=test_generator.generate(
        #             batch_size, num_classes),
        #         validation_steps=valid_steps,
        #         verbose=verb,
        #         callbacks=callback_list
        #     )

    # if num_samples_test == None:
    #     score = model.evaluate_generator(
    #             generator=test_generator.generate(batch_size, num_classes),
    #             steps=test_generator.get_num_samples() / batch_size)
    # else:
    score = model.evaluate_generator(
        generator=test_generator.generate(batch_size, num_classes),
        steps=num_samples_test / batch_size)


    bot_save(model, encoder, where_save_bot)
    # Отрисовка графика результата
    print("Test score: ", score[0])
    print("Test accuracy: ", score[1])

    plt.subplot(211)
    plt.title("Accuracy")
    plt.plot(history.history["accuracy"], color="g", label="Train")
    plt.plot(history.history["val_accuracy"], color="b", label="Validation")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="g", label="Train")
    plt.plot(history.history["val_loss"], color="b", label="Validation")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = '//home//nail//Code_Go//dlgo//data'
    num_games = 1000
    learning_rate = 0.000001
    percent_validation = 30
    file_num_samples = data_dir+'//file_num_samples.txt'
    nsample = open(file_num_samples, 'r')
    try:
        num_samples = nsample.readline()
        num_samples = int(num_samples)
        num_samples_test = nsample.readline()
        num_samples_test = int(num_samples_test)
        nsample.close()
    except:
        num_samples = None
        num_samples_test = None
#  seed используется для генерации случайной выборки игр из всех доступных игр полученных с сервера KGS.
#  используется только в случае подговтоки данных для обучения и не участвует в самом обучении.
#  В книге значение было постоянным и равнялась 1377.
    #seed = random.randint(1,10000000)
    seed = 1377

    epochs = 500
    batch_size = 128
    #optimizer = 'adagrad'
    optimizer = 'adadelta'
    #optimizer = 'SGD'
    patience = 100

    name_model = 'AlphaGo_alphago'
    saved_model = r'../checkpoints/'+str(num_games)+'_'+name_model+'_'+ \
                str(batch_size)+'_bsize_model_epoch_{epoch:3d}_{val_loss:.4f}_{val_accuracy:.4f}.h5'
    saved_model.replace('  ','')
    saved_bot = r'../checkpoints/'+str(num_games)+'_'+name_model+'_bot.h5'
    saved_bot.replace(' ','')

    network_alphago = 'n'
    num_layers = 12
    num_filters = 192
    first_kernel_size = 5
    other_kernel_size = 3

    pr_kgs = input('Only_KGS? (Y/N) ')
    pr_kgs = pr_kgs.lower()
    if pr_kgs == 'n':
        network_alphago = input('Network is AlphaGo(Y/N) ')
        network_alphago = network_alphago.lower()

    if network_alphago == 'y':
        num_layers = int(input('Num_Layers AlphaGo = '))
        num_filters = int(input('Num_filters AlphaGo = '))
        first_kernel_size = int(input('First_kernel_size = ')),
        other_kernel_size = int(input('Other_kernel_size = '))

    if pr_kgs == 'y':
        cont_train = True
        verb = 2
        lst_files = os.listdir(data_dir)
        lst_npy=[]
        for entry in lst_files:
            if fnmatch.fnmatch(entry, '*npy'):
               lst_npy.append(entry)
        cnt_files_begin = len(lst_npy)
        print('---------------------------  Count files npy Before = ', cnt_files_begin)
        # runstr='my_first_network(cont_train, num_games, epochs, batch_size, optimizer, patience, saved_model,'+ \
        #                  'saved_bot, pr_kgs, seed)'
        # cProfile.run(runstr)
        my_first_network(cont_train, num_games, num_samples, num_samples_test,
                         epochs, batch_size, percent_validation, optimizer, learning_rate, patience, saved_model,
                         saved_bot, pr_kgs, seed, name_model,
                         num_layers=num_layers,
                         num_filters=num_filters,
                         first_kernel_size=first_kernel_size,
                         other_kernel_size=other_kernel_size
                         )

        lst_files = os.listdir(data_dir)
        lst_npy = []
        for entry in lst_files:
            if fnmatch.fnmatch(entry, '*npy'):
                lst_npy.append(entry)
        cnt_files_after = len(lst_npy)

        print('----------------------------  Count files npy Before = ', cnt_files_begin)
        print('============================   Count files npy After = ', cnt_files_after)
        print('---- ------------------------------------------------------New files = ',
              cnt_files_after - cnt_files_begin)
        print('Only formed KGS files')

    else:
        # ==================================================
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        # ==================================================

        cont = input("Continue train(C)/from begin(B)? ")
        verb = int(input('How show training process 1 or 2 ?:'))

        if not (verb == 1 or verb == 2):
            print('Wrong parameter for verbose =', verb)
            verb = 2

        if cont.lower() == 'c':
            cont_train = True
        if cont.lower() == 'b':
            cont_train = False



        my_first_network(cont_train, num_games, num_samples, num_samples_test,
                         epochs, batch_size,percent_validation, optimizer, learning_rate, patience,
                         saved_model, saved_bot,pr_kgs,seed, name_model,
                         num_layers,
                         num_filters,
                         first_kernel_size,
                         other_kernel_size
                         )
        sess.close()
