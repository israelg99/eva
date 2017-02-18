from keras.layers import Lambda, Convolution2D
from eva.layers.shift_down import ShiftDown


class FeedVertical(object):
    def __init__(self, filters):
        self.filters = filters

    def __call__(self, model):
        # model = Lambda(ShiftDown)(model)
        model = ShiftDown(model)
        model = Convolution2D(2*self.filters, 1, 1, border_mode='valid')(model)
        return model
