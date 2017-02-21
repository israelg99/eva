from keras.layers import Activation, Merge, Convolution1D, AtrousConvolution1D

from eva.layers.causal_atrous_convolution1d import CausalAtrousConvolution1D


class WavenetBlock(object):
    def __init__(self, filters, index):
        self.filters = filters
        self.index = index

    def __call__(self, model):
        original = model

        atrous_rate = 2**self.index
        if self.index > 9:
            print('Since atrous rates are exponential, more than 9 is not recommended.')
            print('It is advised to split the stacks with no more than 9 blocks for each stack.')

        tanh_out = CausalAtrousConvolution1D(self.filters, 2, atrous_rate=atrous_rate, border_mode='valid')(model)
        tanh_out = Activation('tanh')(tanh_out)

        sigm_out = CausalAtrousConvolution1D(self.filters, 2, atrous_rate=atrous_rate, border_mode='valid')(model)
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
        skips = [None] * self.depth
        for _ in range(self.stacks):
            for i in range(self.depth):
                res, skip = WavenetBlock(self.filters, i)(res)
                skips[i] = skip

        return res, skips
