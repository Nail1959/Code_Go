# tag::alphago_base[]
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D


def alphago_model(input_shape,   # <1>
                  num_filters=192,  # <2>
                  num_layers = 12,
                  first_kernel_size=5,
                  other_kernel_size=3):  # <3>

    model = Sequential()
    kernel_size = first_kernel_size
    model.add(
        Conv2D(num_filters, kernel_size, input_shape=input_shape, padding='same',
               data_format='channels_first', activation='relu'))

    kernel_size = other_kernel_size
    for i in range(2, num_layers):  # <4> Nail 12  --> num_layers
        model.add(
            Conv2D(num_filters, kernel_size, padding='same',
                   data_format='channels_first', activation='relu'))


        model.add(Flatten())

        return model
# end::alphago_policy[]

