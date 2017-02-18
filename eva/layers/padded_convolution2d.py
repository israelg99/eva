from keras.layers import Convolution2D, ZeroPadding2D


class PaddedConvolution2D(object):
    def __init__(self, filters, filter_size, stack, mask='B'):
        self.filters = filters
        self.filter_size = filter_size
        self.stack = stack
        self.mask = mask

    def __call__(self, model):
        if self.stack == 'vertical':
            model = ZeroPadding2D(padding=(self.filter_size[0]//2, 0, self.filter_size[1]//2, self.filter_size[1]//2))(model)
            model = Convolution2D(2*self.filters, self.filter_size[0]//2+1, self.filter_size[1], border_mode='valid')(model)
        elif self.stack == 'horizontal':
            model = ZeroPadding2D(padding=(0, 0, self.filter_size[1]//2, 0))(model)
            if self.mask == 'A':
                model = Convolution2D(2*self.filters, 1, self.filter_size[1]//2, border_mode='valid')(model)
            else:
                model = Convolution2D(2*self.filters, 1, self.filter_size[1]//2+1, border_mode='valid')(model)

        return model
