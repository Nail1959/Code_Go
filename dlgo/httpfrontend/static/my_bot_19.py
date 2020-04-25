# -*- coding: utf-8 -*-
import h5py
from keras.models import Sequential
from keras.layers import Dense
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.networks import large
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#==================================================
#

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)
#==================================================

def fit_model(pathdir = r"../checkpoints/deep_bot.h5" ):
    go_board_rows, go_board_cols = 19, 19
    nb_classes = go_board_rows * go_board_cols
    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())
    X, y = processor.load_go_data(num_samples=50)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    model = Sequential()
    network_layers = large.layers(input_shape)
    for layer in network_layers:
        model.add(layer)

    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                    metrics=['accuracy'])
    model.fit(X, y, batch_size=128, epochs=3)

    deep_learning_bot = DeepLearningAgent(model, encoder)
    model_file = h5py.File(pathdir, "w")
    deep_learning_bot.serialize(model_file)

    return

def bot_web_play(pathdir = r"/home/nail/CODE_GO/dlgo/httpfrontend/static/bot_19.h5"):

    model_file = h5py.File(pathdir, "r")
    bot_from_file = load_prediction_agent(model_file)
    bot_map = {'predict': bot_from_file}
    web_app = get_web_app(bot_map)
    web_app.run(host='localhost', port=5001, debug=True)

if __name__  ==  "__main__":
    pdir = r"/home/nail/CODE_GO/dlgo/httpfrontend/static/bot_19.h5"
    #fit_model(pdir)
    print('==================================================Train is over ')
    bot_web_play(pdir)
