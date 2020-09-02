import h5py
from dlgo.agent.predict import  load_prediction_agent
from dlgo.rl import load_q_agent
from dlgo.httpfrontend import get_web_app
import tensorflow as tf
import os

import sys
sys.setrecursionlimit(10000)

from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#==================================================
#
#
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
#==================================================
pdir = r"../checkpoints/"


pf = input('Bot Name: ')
ph = pdir + pf + '.h5'
model_file = h5py.File(ph, "r")


bot_from_file = load_prediction_agent(model_file)
web_app = my_get_web_app({'predict': bot_from_file})
web_app.run(threaded=False)


sess.close()