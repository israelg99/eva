from keras.layers import Convolution2D, Merge
from keras.layers.advanced_activations import PReLU

from eva.layers.masked_convolution2d import MaskedConvolution2D

class ResidualBlock(object):
    def __init__(self, filters):
        self.filters = filters

    def __call__(self, model):
        # 2h -> h
        block = PReLU()(model)
        block = MaskedConvolution2D(self.filters//2, 1, 1)(block)

        # h 3x3 -> h
        block = PReLU()(block)
        block = MaskedConvolution2D(self.filters//2, 3, 3, border_mode='same')(block)

        # h -> 2h
        block = PReLU()(block)
        block = MaskedConvolution2D(self.filters, 1, 1)(block)

        return Merge(mode='sum')([model, block])

class ResidualBlockList(object):
    def __init__(self, filters, length):
        self.filters = filters
        self.length = length

    def __call__(self, model):
        for _ in range(self.length):
            model = ResidualBlock(self.filters)(model)

        return model