from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, Flatten, Dense, Reshape, Lambda
from keras.layers.advanced_activations import PReLU
from keras.engine.topology import merge
from keras.optimizers import Nadam
import keras.backend.tensorflow_backend as K

from eva.layers.residual_block import ResidualBlockList
from eva.layers.masked_convolution2d import MaskedConvolution2D

def PixelCNN(input_shape, filters, blocks, build=True):
    width, height, channels = input_shape

    # TODO: Make it scalable to any amount of channels.
    input_map = Input(shape=input_shape, name='input_map')

    model = MaskedConvolution2D(filters, 7, 7, mask='A', border_mode='same', name='masked2d_A')(input_map)

    model = ResidualBlockList(filters, blocks)(model)
    model = PReLU()(model)

    model = MaskedConvolution2D(filters, 1, 1)(model)
    model = PReLU()(model)

    model = MaskedConvolution2D(3*256, 1, 1, name='channels_mult_palette')(model)
    model = Reshape((input_shape[0], input_shape[1], 256, input_shape[2]), name='palette_channels')(model)

    # TODO: Make it scalable to any amount of channels.

    red = Lambda(lambda x: x[:, :, :, :, 0], name='red_extract')(model)
    red = Reshape((input_shape[0] * input_shape[1], 256), name='hw_red-palette')(red)
    red = Activation('softmax', name='red')(red)

    green = Lambda(lambda x: x[:, :, :, :, 1], name='green_extract')(model)
    green = Reshape((input_shape[0] * input_shape[1], 256), name='hw_green-palette')(green)
    green = Activation('softmax', name='green')(green)

    blue = Lambda(lambda x: x[:, :, :, :, 2], name='blue_extract')(model)
    blue = Reshape((input_shape[0] * input_shape[1], 256), name='hw_blue-palette')(blue)
    blue = Activation('softmax', name='blue')(blue)

    # TODO: Make is scalable to any amount of channels.

    if build:
        model = Model(input=input_map, output=[red, green, blue])
        model.compile(optimizer=Nadam(),
                      loss={    'red':   'sparse_categorical_crossentropy',
                                'green': 'sparse_categorical_crossentropy',
                                'blue':  'sparse_categorical_crossentropy'})

    return model
