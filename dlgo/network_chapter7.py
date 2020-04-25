from __future__ import print_function
from __future__ import absolute_import

from os import sys

from multiprocessing import Process, freeze_support



#sys.path.append("E:\\Proj_GO\\CODE_GO")  # Добавил чтобы модули были видны из папки.



from dlgo.data.parallel_processor import GoDataProcessor
#from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.networks import large
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint

def my_first_network():
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 100
#    encoder = OnePlaneEncoder((go_board_rows, go_board_cols))
    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())

    generator = processor.load_go_data('train', num_games, use_generator=True)
    test_generator = processor.load_go_data('test', num_games, use_generator=True)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    network_layers = large.layers(input_shape)
    model = Sequential()
    for layer in network_layers:
        model.add(layer)

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                        metrics=['accuracy'])

    epochs = 10
    batch_size = 128
    model.fit_generator(
            generator=generator.generate(batch_size, num_classes),
            epochs=epochs,
            steps_per_epoch=generator.get_num_samples() / batch_size,
            validation_data=test_generator.generate(
                                batch_size, num_classes),
            validation_steps=test_generator.get_num_samples() / batch_size,
            callbacks=[
                ModelCheckpoint('../checkpoints/large_model_epoch_{epoch}.h5')
            ])

    model.evaluate_generator(
    generator=test_generator.generate(batch_size, num_classes),
    steps=test_generator.get_num_samples() / batch_size)

if __name__ == "__main__":
    freeze_support()
    my_first_network()