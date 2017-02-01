from keras import backend as K
from keras.layers import Layer


class ColorExtract(Layer):
    """ TODO: Make is scalable to any amount of channels. """
    def __init__(self, channel, **kwargs):
        super().__init__(**kwargs)
        assert channel in (0, 1, 2)
        self.channel = channel

    def call(self, x, mask=None):
        return x[:, :, :, (self.channel*256):(self.channel+1)*256]

    def get_output_shape_for(self, input_shape):
        output = list(input_shape)
        return (output[0], output[1], output[2], 256)

    def get_config(self):
        return dict(list(super().get_config().items()) + list({'channel': self.channel}.items()))
