from keras.models import Model
from keras.layers import Input, Convolution1D, Activation, Merge, Lambda
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam

from eva.layers.causal_atrous_convolution1d import CausalAtrousConvolution1D
from eva.layers.wavenet_block import WavenetBlock, WavenetBlocks


def Wavenet(input_shape, filters, depth, stacks, last=0, h=None, build=True):
    # TODO: Soft targets? A float to make targets a gaussian with stdev.
    # TODO: Train only receptive field. The temporal-first outputs are computed from zero-padding.
    # TODO: Global conditioning?
    # TODO: Local conditioning?

    _, nb_bins = input_shape

    input_audio = Input(input_shape, name='audio_input')

    model = CausalAtrousConvolution1D(filters, 2, mask_type='A', atrous_rate=1, border_mode='valid')(input_audio)

    out, skip_connections = WavenetBlocks(filters, depth, stacks)(model)

    out = Merge(mode='sum', name='merging_skips')(skip_connections)
    out = PReLU()(out)

    out = Convolution1D(nb_bins, 1, border_mode='same')(out)
    out = PReLU()(out)

    out = Convolution1D(nb_bins, 1, border_mode='same')(out)

    # https://storage.googleapis.com/deepmind-live-cms/documents/BlogPost-Fig2-Anim-160908-r01.gif
    if last > 0:
        out = Lambda(lambda x: x[:, -last:], output_shape=(last, out._keras_shape[2]), name='last_out')(out)

    out = Activation('softmax')(out)

    if build:
        model = Model(input_audio, out)
        model.compile(Nadam(), 'sparse_categorical_crossentropy')

    return model

def compute_receptive_field(sample_rate, depth, stacks):
    receptive_field = stacks * (2 ** depth) - (stacks - 1)
    receptive_field_ms = (receptive_field * 1000) / sample_rate
    return receptive_field, receptive_field_ms