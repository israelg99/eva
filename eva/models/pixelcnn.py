from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, Flatten, Dense
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam

from eva.layers.residual_block import ResidualBlockList
from eva.layers.masked_convolution2d import MaskedConvolution2D

def PixelCNN(input_shape, filters, blocks, softmax=False, build=True):
    input_map = Input(shape=input_shape)

    model = MaskedConvolution2D(filters, 7, 7, mask='A', border_mode='same')(input_map)
    model = PReLU()(model)

    model = ResidualBlockList(model, filters, blocks)

    model = Convolution2D(filters//2, 1, 1)(model)
    model = PReLU()(model)

    model = Convolution2D(filters//2, 1, 1)(model)
    model = PReLU()(model)

    model = Convolution2D(1, 1, 1)(model)

    if not softmax:
        model = Activation('sigmoid')(model)
    else:
        raise NotImplementedError()

    if build:
        model = Model(input=input_map, output=model)
        model.compile(loss='binary_crossentropy',
                      optimizer=Nadam(),
                      metrics=['accuracy', 'fbeta_score', 'matthews_correlation'])

    return model
