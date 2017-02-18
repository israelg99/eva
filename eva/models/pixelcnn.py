from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, Flatten, Dense, Reshape, Lambda
from keras.layers.advanced_activations import PReLU
from keras.engine.topology import merge
from keras.optimizers import Nadam
import keras.backend.tensorflow_backend as K

from eva.layers.residual_block import ResidualBlockList
from eva.layers.masked_convolution2d import MaskedConvolution2D

def PixelCNN(input_shape, filters, blocks, build=True):
    width, height, channels = input_shape

    input_map = Input(shape=input_shape, name='input_map')

    model = MaskedConvolution2D(filters, 7, 7, mask='A', border_mode='same', name='masked2d_A')(input_map)

    model = ResidualBlockList(filters, blocks)(model)
    model = PReLU()(model)

    model = MaskedConvolution2D(filters, 1, 1)(model)
    model = PReLU()(model)

    # TODO: Make it scalable to any amount of palette.
    if channels == 1:
        outs = Convolution2D(1, 1, 1, activation='sigmoid', border_mode='valid')(model)
    else:
        model = MaskedConvolution2D(channels*256, 1, 1, name='channels_mult_palette')(model)
        model = Reshape((input_shape[0], input_shape[1], 256, input_shape[2]), name='palette_channels')(model)

        outs = [None] * channels
        for i in range(channels):
            outs[i] = Lambda(lambda x: x[:, :, :, :, 0], name='channel'+str(i)+'_extract')(out)
            outs[i] = Reshape((input_shape[0] * input_shape[1], 256), name='hw_palette'+str(i))(outs[i])
            outs[i] = Activation('softmax', name='channel'+str(i))(outs[i])

    if build:
        model = Model(input=input_img, output=outs)
        model.compile(optimizer=Nadam(), loss='binary_crossentropy' if channels == 1 else 'sparse_categorical_crossentropy')

    return model
