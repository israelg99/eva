from keras.layers import Convolution2D, Merge
from keras.layers.advanced_activations import PReLU

from eva.layers.masked_convolution2d import MaskedConvolution2D

def ResidualBlock(model, filters):
    # 2h -> h
    block = PReLU()(model)
    block = MaskedConvolution2D(filters//2, 1, 1)(block)

    # h 3x3 -> h
    block = PReLU()(block)
    block = MaskedConvolution2D(filters//2, 3, 3, border_mode='same')(block)

    # h -> 2h
    block = PReLU()(block)
    block = MaskedConvolution2D(filters, 1, 1)(block)

    return Merge(mode='sum')([model, block])

def ResidualBlockList(model, filters, length):
    for _ in range(length):
        model = ResidualBlock(model, filters)

    return model
