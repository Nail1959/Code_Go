#-*- coding:UTF-8 -*-
from dlgo.encoders.simple import SimpleEncoder
from keras.models import load_model
import h5py
from dlgo.agent.predict import DeepLearningAgent

def bot_save(model, encoder, where_save_bot):
    # Сохранение для бота чтобы играть в браузере с ботом play_predict_19.html
    deep_learning_bot = DeepLearningAgent(model, encoder)
    model_file = h5py.File(where_save_bot, "w")
    deep_learning_bot.serialize(model_file)

if __name__ == '__main__':
    name_bot = input('Имя бота для сохранения = ')
    name_model = input('Имя обученной модели для создания бота = ')
    name_model = r'../checkpoints/' + name_model+'.h5'
    saved_bot = r'../checkpoints/' + name_bot + '_bot.h5'
    encoder = SimpleEncoder((19, 19))

    model = load_model(filepath=name_model)
    bot_save(model, encoder, saved_bot)