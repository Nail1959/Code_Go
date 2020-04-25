import h5py
from keras.models import Sequential
from keras.layers import Dense
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.networks import large
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import os
from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#==================================================
#

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
grph = tf.compat.v1.get_default_graph
sess = tf.compat.v1.Session(config=config)
set_session(sess)

#==================================================
pdir = r"/home/nail/CODE_GO/dlgo/httpfrontend/static/bot_19.h5"



train = True
if train is True:
    go_board_rows, go_board_cols = 19, 19
    nb_classes = go_board_rows * go_board_cols
    encoder = OnePlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())
    X, y = processor.load_go_data(num_samples=10)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    model = Sequential()
    network_layers = large.layers(input_shape)
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad',
                  metrics=['accuracy'])
    model.fit(X, y, batch_size=128, epochs=2,verbose=2)
    # ?????????? ?????? ??? ????.
    deep_learning_bot = DeepLearningAgent(model, encoder)
    model_file = h5py.File(pdir, "w")
    deep_learning_bot.serialize(model_file)


else:
    print(u'?????? ?? ??????')

