from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, Flatten, Dense
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam

from eva.models.residual_block import ResidualBlockList
from eva.layers.masked_convolution2d import MaskedConvolution2D

def PixelCNN(input_shape, filters, blocks, softmax=False, build=True):
    input_map = Input(shape=input_shape)

    model = MaskedConvolution2D(filters, 7, 7, mask='A', border_mode='same')(input_map)

    model = ResidualBlockList(model, filters, blocks)
    model = PReLU()(model)

    model = MaskedConvolution2D(filters, 1, 1)(model)

    # How the fuck are we aligning it for output activation and activating it?
    # TODO: Everything from here has to be changed.
    model = Flatten()(model)

    # Discrete? ummm..
    discrete = Dense(input_shape[0] * input_shape[1] * 256)
    del discrete # Whops.

    # Continuous? Regressive?
    continuous = Dense(input_shape[0] * input_shape[1])
    model = continuous(model)

    if not softmax:
        model = Activation('sigmoid')(model)
    else:
        raise NotImplementedError()

    if build:
        model = Model(input=input_map, output=model)
        model.compile(loss='mse',
                      optimizer=Nadam(),
                      metrics=['accuracy'])

    return model
