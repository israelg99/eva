import math

import numpy as np

from keras import backend as K
from keras.layers import Convolution2D

class MaskedConvolution2D(Convolution2D):
    def __init__(self, *args, mask='B', **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_type = mask

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

        for j1 in range(self.mask.shape[2]):
            for j2 in range(self.mask.shape[3]//self.mask.shape[2]):
                for j3 in range(self.mask.shape[2]):
                    if op(j1, j3):
                        self.mask[math.floor(filter_center), math.floor(filter_center),
                                  j1, j2*self.mask.shape[2]+j3] = 0

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
