import keras.backend as K
from keras.layers import AtrousConvolution1D
from keras.utils.np_utils import conv_output_length

class CausalAtrousConvolution1D(AtrousConvolution1D):
    def __init__(self, *args, mask_type='B', **kwargs):
        super().__init__(*args, **kwargs)
        if self.border_mode != 'valid':
            raise ValueError("Causal mode dictates border_mode=valid.")
        self.length = self.atrous_rate * (self.filter_length - 1)

        # XXX debug.
        if mask_type == 'A':
            self.length += 1

    def get_output_shape_for(self, input_shape):
        length = conv_output_length(input_shape[1] + self.length,
                                    self.filter_length,
                                    self.border_mode,
                                    self.subsample[0],
                                    dilation=self.atrous_rate)

        return (input_shape[0], length, self.nb_filter)

    def call(self, x, mask=None):
        return super().call(K.asymmetric_temporal_padding(x, self.length, 0), mask)
