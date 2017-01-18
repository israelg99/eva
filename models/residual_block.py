from keras.layers import Convolution2D
from keras.layers.advanced_activations import PReLU

from eva.layers.masked_convolution2d import MaskedConvolution2D

def ResidualBlock(model, filters):
    # 2h -> h
    model = Convolution2D(filters//2, 1, 1)(model)
    model = PReLU()(model)

    # h 3x3 -> h
    model = MaskedConvolution2D(filters//2, 3, 3, border_mode='same')(model)
    model = PReLU()(model)

    # h -> 2h
    model = Convolution2D(filters, 1, 1)(model)
    model = PReLU()(model)

    return model

def ResidualBlockList(model, filters, length):
    for _ in range(length):
        model = ResidualBlock(model, filters)

    return model
