from keras.layers import Convolution2D, Reshape, Lambda, Activation

from eva.layers.masked_convolution2d import MaskedConvolution2D

class OutChannels(object):
    def __init__(self, height, width, channels, masked=False, palette=256):
        self.height = height
        self.width = width
        self.channels = channels
        self.cxp = MaskedConvolution2D if masked else Convolution2D
        self.palette = palette

    def __call__(self, model):
        if self.channels == 1:
            outs = Convolution2D(1, 1, 1, border_mode='valid')(model)
            outs = Activation('sigmoid')(outs)
        else:
            out = self.cxp(self.palette*self.channels, 1, 1, border_mode='valid', name='channels_mult_palette')(model)
            out = Reshape((self.height, self.width, self.palette, self.channels), name='palette_channels')(out)

            outs = [None] * self.channels
            for i in range(self.channels):
                outs[i] = Lambda(lambda x: x[:, :, :, :, i], name='channel'+str(i)+'_extract')(out)
                outs[i] = Reshape((self.height * self.width, self.palette), name='hw_palette'+str(i))(outs[i])
                outs[i] = Activation('softmax', name='channel'+str(i))(outs[i])

        return outs