from keras.layers import Convolution2D, Merge
import keras.backend as K

from eva.layers.padded_convolution2d import PaddedConvolution2D
from eva.layers.feed_vertical import FeedVertical
from eva.layers.gated_block import GatedBlock


class GatedCNN(object):
    def __init__(self, filters, h=None):
        self.filters = filters
        self.h = h

    # def __call__(self, model):
    #     v_model = PaddedConvolution2D(self.filters, 7, 'vertical')(model)
    #     feed_vertical = FeedVertical(self.filters)(v_model)
    #     v_model = GatedBlock(self.filters, h=self.h)(v_model)

    #     h_model = PaddedConvolution2D(self.filters, 7, 'horizontal', 'A')(model)
    #     h_model = GatedBlock(self.filters, 'horizontal', v_map=feed_vertical, h=self.h, crop_right=True)(h_model)
    #     h_model = Convolution2D(self.filters, 1, 1, border_mode='valid')(h_model)

    #     return (v_model, h_model)

    # def __call__(self, v_model, h_model):
    #     v_model = PaddedConvolution2D(self.filters, 7, 'vertical')(v_model)
    #     feed_vertical = FeedVertical(self.filters)(v_model)
    #     v_model = GatedBlock(self.filters, h=self.h)(v_model)

    #     h_model_new = PaddedConvolution2D(self.filters, 7, 'horizontal', 'A')(h_model)
    #     h_model_new = GatedBlock(self.filters, 'horizontal', v_map=feed_vertical, h=self.h, crop_right=True)(h_model_new)
    #     h_model_new = Convolution2D(self.filters, 1, 1, border_mode='valid')(h_model_new)
    #     h_model = Merge(mode='sum')([h_model_new, h_model])

    #     return (v_model, h_model)

    def __call__(self, model1, model2=None):
        if model2 is None:
            h_model = model1
            filter_size = (7, 7)
        else:
            h_model = model2
            filter_size = (3, 3)

        v_model = PaddedConvolution2D(self.filters, filter_size, 'vertical')(model1)
        feed_vertical = FeedVertical(self.filters)(v_model)
        v_model = GatedBlock(self.filters, h=self.h)(v_model)

        h_model_new = PaddedConvolution2D(self.filters, filter_size, 'horizontal', 'A')(h_model)
        h_model_new = GatedBlock(self.filters, v=feed_vertical, h=self.h, crop_right=True)(h_model_new)
        h_model_new = Convolution2D(self.filters, 1, 1, border_mode='valid')(h_model_new)

        return (v_model, h_model_new if model2 is None else Merge(mode='sum')([h_model_new, h_model]))

class GatedCNNs(object):
    def __init__(self, filters, length, h=None):
        self.filters = filters
        self.length = length
        self.h = h

    def __call__(self, v_model, h_model):
        for _ in range(self.length):
            v_model, h_model = GatedCNN(self.filters, self.h)(v_model, h_model)

        return h_model