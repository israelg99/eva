from keras.layers import Activation, Merge, Convolution1D, AtrousConvolution1D

from eva.layers.causal_atrous_convolution1d import CausalAtrousConvolution1D


class WavenetBlock(object):
    def __init__(self, filters, rate):
        self.filters = filters
        self.rate = rate

    def __call__(self, model):
        original = model

        tanh_out = CausalAtrousConvolution1D(self.filters, 2, atrous_rate=self.rate, border_mode='valid')(model)
        tanh_out = Activation('tanh')(tanh_out)

        sigm_out = CausalAtrousConvolution1D(self.filters, 2, atrous_rate=self.rate, border_mode='valid')(model)
        sigm_out = Activation('sigmoid')(sigm_out)

        model = Merge(mode='mul')([tanh_out, sigm_out])

        skip_x = Convolution1D(self.filters, 1, border_mode='same')(model)

        res_x = Convolution1D(self.filters, 1, border_mode='same')(model)
        res_x = Merge(mode='sum')([original, res_x])
        return res_x, skip_x

class WavenetBlocks(object):
    def __init__(self, filters, depth, stacks=1, h=None):
        self.filters = filters
        self.depth = depth
        self.stacks = stacks
        self.h = h

    def __call__(self, res):
        skips = [None] * (self.stacks * self.depth)
        for j in range(self.stacks):
            for i in range(self.depth):
                res, skip = WavenetBlock(self.filters, 2**i)(res)
                skips[j*self.depth + i] = skip

        return res, skips
