from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, Flatten, Dense, Reshape, Lambda
from keras.layers.advanced_activations import PReLU
from keras.engine.topology import merge
from keras.optimizers import Nadam
import keras.backend as K

from eva.layers.residual_block import ResidualBlockList
from eva.layers.masked_convolution2d import MaskedConvolution2D

def PixelCNN(input_shape, filters, blocks, build=True):
    width, height, channels = input_shape

    input_map = Input(shape=input_shape, name='input_map')

    model = MaskedConvolution2D(filters, 7, 7, mask='A', border_mode='same')(input_map)
    model = PReLU()(model)

    model = ResidualBlockList(model, filters, blocks)

    model = MaskedConvolution2D(filters, 1, 1)(model)
    model = PReLU()(model)

    model = MaskedConvolution2D(3*256, 1, 1)(model)
    model = PReLU()(model)

    # TODO: Make it scalable to any amount of channels.

    red = MaskedConvolution2D(256, 1, 1)(model)
    red = Reshape((input_shape[0] * input_shape[1], 256))(red)
    red = Activation('softmax', name='red')(red)

    green = MaskedConvolution2D(256, 1, 1)(model)
    green = Reshape((input_shape[0] * input_shape[1], 256))(green)
    green = Activation('softmax', name='green')(green)

    blue = MaskedConvolution2D(256, 1, 1)(model)
    blue = Reshape((input_shape[0] * input_shape[1], 256))(blue)
    blue = Activation('softmax', name='blue')(blue)

    # TODO: Make is scalable to any amount of channels.

    if build:
        model = Model(input=input_map, output=[red, green, blue])
        model.compile(optimizer=Nadam(clipnorm=1., clipvalue=1.),
                      loss={    'red':   'sparse_categorical_crossentropy',
                                'green': 'sparse_categorical_crossentropy',
                                'blue':  'sparse_categorical_crossentropy'})

    return model
