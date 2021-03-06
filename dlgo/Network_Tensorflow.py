#-*- coding:UTF-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.my_sevenplane_s import MySevenPlaneEncoder_S
from dlgo.networks import my_network

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from dlgo.agent.predict import DeepLearningAgent
import os
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import math
import glob
import random
import cProfile

#from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.disable_v2_behavior()
#==================================================
#
#tf.compat.v1.disable_eager_execution()
#config = tf.compat.v1.ConfigProto()
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
config.log_device_placement = True
#g = tf.Graph()
#set_session(tf.compat.v1.Session(config=config))
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
# sess.run(tf.compat.v1.global_variables_initializer())
#==================================================
# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def bot_save(model, encoder, where_save_bot):
    # Сохранение для бота чтобы играть в браузере с ботом play_predict_19.html
    deep_learning_bot = DeepLearningAgent(model, encoder)
    model_file = h5py.File(where_save_bot, "w")
    deep_learning_bot.serialize(model_file)


def my_first_network(cont_train=True, num_games=100, epochs=10, batch_size=128,
                     optimizer='adadelta', patience=5,
                     where_save_model = '../checkpoints/large_model_epoch_{epoch:3d}_{val_loss:.3f}_{val_accuracy:.3f}.h5',
                     where_save_bot='../checkpoints/deep_bot.h5',pr_kgs='n', seed =1337):
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols


    encoder = MySevenPlaneEncoder_S((go_board_rows,go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name(), data_directory='data')

    if pr_kgs == 'y':  # Only forming train and test data   into data directory

        generator = processor.load_go_data('train', num_games, use_generator=True, seed = seed)
        test_generator = processor.load_go_data('test', num_games, use_generator=True,seed = seed)
        return

    else:
        generator = processor.load_go_data('train', num_games, use_generator=True, seed=0)
        test_generator = processor.load_go_data('test', num_games, use_generator=True,seed=0)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    network_layers = my_network.layers(input_shape)

    train_log = 'training_'+str(num_games)+'_epochs_'+str(epochs)+'_'+optimizer+'.log'
    csv_logger = CSVLogger(train_log, append=True, separator=';')
    if patience > 2:
        r_patience = patience - 1
    else:
        r_patience = patience
    Reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=r_patience,
                               verbose=1, mode='auto', min_delta=0.0005, cooldown=0, min_lr=0)

    tensor_board_log_dir='/home/nail/CODE_GO/checkpoints/my_log_dir'
    tensorboard = TensorBoard(log_dir=tensor_board_log_dir,histogram_freq=1,embeddings_freq=1,write_graph=True)

    if optimizer == 'adagrad':
        callback_list = [ModelCheckpoint(where_save_model,
                                save_best_only=True),
                         EarlyStopping(monitor='val_accuracy', mode='auto',verbose=verb, patience=patience,
                                       min_delta=0,restore_best_weights=True),
                         csv_logger,
                         Reduce,
                         tensorboard
                          ]
    else:
        callback_list = [ModelCheckpoint(where_save_model,
                                save_best_only=True),
                         EarlyStopping(monitor='val_accuracy', mode='auto', verbose=verb, patience=patience,
                                       min_delta=0,restore_best_weights=True),
                         csv_logger,
                         Reduce,
                         tensorboard
                          ]

    if cont_train is False: # Обучение с самого начала с случайных весов
        model = Sequential()
        for layer in network_layers:
            model.add(layer)

        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        #model.summary()

        history = model.fit_generator(
            generator=generator.generate(batch_size, num_classes),
            epochs=epochs,
            steps_per_epoch=generator.get_num_samples() / batch_size,
            validation_data=test_generator.generate(
                batch_size, num_classes),
            validation_steps=test_generator.get_num_samples() / batch_size,
            verbose=verb,
            callbacks= callback_list
            )

    if cont_train is True: # Обучение используя уже предобученную модель, продолжение обучения.

        model = load_model('../checkpoints/model_continue.h5')
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
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

    score = model.evaluate_generator(
            generator=test_generator.generate(batch_size, num_classes),
            steps=test_generator.get_num_samples() / batch_size)


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
    num_games = 10000
    seed = random.randint(1,10000000)

    epochs = 1
    batch_size = 128
    optimizer = 'adagrad'
    patience = 3

    saved_model = r'../checkpoints/'+str(num_games)+'_large_model_epoch_{epoch:3d}_{val_loss:.4f}_{val_accuracy:.4f}.h5'
    saved_bot = r'../checkpoints/'+str(num_games)+'_deep_bot.h5'

    pr_kgs = input('Only_KGS? (Y/N) ')
    pr_kgs = pr_kgs.lower()
    if pr_kgs=='y':

        cont_train = True
        verb =2
        cnt_files_begin = sum(os.path.isfile(f) for f in glob.glob('/home/nail/CODE_GO/dlgo/data/*.npy'))
        print('---------------------------  Count files npy Before = ', cnt_files_begin)
        runstr='my_first_network(cont_train, num_games, epochs, batch_size, optimizer, patience, saved_model,'+ \
                         'saved_bot, pr_kgs, seed)'
        cProfile.run(runstr)
        # my_first_network(cont_train, num_games, epochs, batch_size, optimizer, patience, saved_model,
        #                  saved_bot, pr_kgs, seed)
        cnt_files_after = sum(os.path.isfile(f) for f in glob.glob('/home/nail/CODE_GO/dlgo/data/*.npy'))
        print('============================   Count files npy After = ', cnt_files_after)
        print('---- ------------------------------------------------------New files = ',
              cnt_files_after - cnt_files_begin)
        print('Only formed KGS files')

    else:

        cont = input("Continue train(C)/from begin(B)? ")
        verb = int(input('How show training process 1 or 2 ?:'))
        if not (verb == 1 or verb == 2):
            print('Wrong parameter for verbose =', verb)
            verb = 2

        if cont.lower() == 'c':
            cont_train = True
        if cont.lower() == 'b':
            cont_train = False

        my_first_network(cont_train, num_games, epochs, batch_size,optimizer, patience, saved_model, saved_bot,pr_kgs,seed)

        sess.close()