import math

import numpy as np

from keras import backend as K
from keras.layers import Convolution2D

class MaskedConvolution2D(Convolution2D):
    def __init__(self, *args, mask='B' , n_channels=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_type = mask
        self.n_channels = n_channels

        # TODO: Define mask here? we can use filter_size and number of filters to predict the weights shape, however, build() is still a safer place for it.
        self.mask = None

    def build(self, input_shape):
        super().build(input_shape)

        self.mask = np.ones(self.W_shape)

        filter_size = self.mask.shape[0]
        filter_center = filter_size / 2

        self.mask[math.ceil(filter_center):] = 0
        self.mask[math.floor(filter_center):, math.ceil(filter_center):] = 0

        op = np.greater_equal if self.mask_type == 'A' else np.greater

        # for i in range(self.n_channels):
        #     for j in range(self.n_channels):
        #         if op(i, j):
        #             self.mask[math.floor(filter_center), math.floor(filter_center), i::self.n_channels, j::self.n_channels] = 0

        if self.mask_type == 'A':
            self.mask[math.floor(filter_center), math.floor(filter_center)] = 0

        self.mask = K.variable(self.mask)

    def call(self, x, mask=None):
        # TODO: learn what is this mask parameter and how to use it.
        output = K.conv2d(x, self.W * self.mask, strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering,
                          filter_shape=self.W_shape)
        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        output = self.activation(output)
        return output

    def get_config(self):
        return dict(list(super().get_config().items()) + list({'mask': self.mask_type}.items()))
