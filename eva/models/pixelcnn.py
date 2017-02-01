from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, Flatten, Dense, Reshape
from keras.layers.advanced_activations import PReLU
from keras.engine.topology import merge
from keras.optimizers import Nadam

from eva.layers.residual_block import ResidualBlockList
from eva.layers.masked_convolution2d import MaskedConvolution2D
from eva.layers.color_extract import ColorExtract

def PixelCNN(input_shape, filters, blocks, build=True):
    width, height, channels = input_shape

    input_map = Input(shape=input_shape, name='input_map')

    model = MaskedConvolution2D(filters, 7, 7, mask='A', border_mode='same')(input_map)
    model = PReLU()(model)

    model = ResidualBlockList(model, filters, blocks)

    model = MaskedConvolution2D(filters, 1, 1)(model)
    model = PReLU()(model)

    model = MaskedConvolution2D(256*3, 1, 1)(model)
    model = PReLU()(model)

    # TODO: Make is scalable to any amount of channels.
    # TODO: SPARSE IT!!!!!

    red = ColorExtract(0)(model)
    red = Reshape((input_shape[0] * input_shape[1], 256))(red)
    red = Activation('softmax', name='red')(red)

    green = ColorExtract(1)(model)
    green = Reshape((input_shape[0] * input_shape[1], 256))(green)
    green = Activation('softmax', name='green')(green)

    blue = ColorExtract(2)(model)
    blue = Reshape((input_shape[0] * input_shape[1], 256))(blue)
    blue = Activation('softmax', name='blue')(blue)

    # TODO: Make is scalable to any amount of channels.
    # TODO: SPARSE IT!!!!!
    if build:
        model = Model(input=input_map, output=[red, green, blue])
        model.compile(optimizer=Nadam(clipnorm=1., clipvalue=1.),
                      loss={    'red':   'categorical_crossentropy',
                                'green': 'categorical_crossentropy',
                                'blue':  'categorical_crossentropy'})

    return model
