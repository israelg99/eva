from keras.layers import Lambda, Dense, Reshape, Merge
import keras.backend as K

from eva.layers.padded_convolution2d import PaddedConvolution2D
from eva.layers.feed_vertical import FeedVertical


class GatedBlock(object):
    def __init__(self, filters, v=None, h=None, crop_right=False):
        self.filters = filters
        self.v = v
        self.h = h
        self.crop_right = crop_right

    def __call__(self, model):
        if self.crop_right:
            model = Lambda(lambda x: x[:, :, :K.int_shape(x)[2]-1, :])(model)

        if self.v is not None:
            model = Merge(mode='sum')([model, self.v])

        if self.h is not None:
            hV = Dense(output_dim=2*self.filters)(self.h)
            hV = Reshape((1, 1, 2*self.filters))(hV)
            model = Lambda(lambda x: x[0]+x[1])([model,hV])

        model_f = Lambda(lambda x: x[:,:,:,:self.filters])(model)
        model_g = Lambda(lambda x: x[:,:,:,self.filters:])(model)

        model_f = Lambda(lambda x: K.tanh(x))(model_f)
        model_g = Lambda(lambda x: K.sigmoid(x))(model_g)

        res = Merge(mode='mul')([model_f, model_g])
        return res
