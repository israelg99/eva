from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, Flatten, Dense, Reshape, Lambda
from keras.layers.advanced_activations import PReLU
from keras.engine.topology import merge
from keras.optimizers import Nadam
import keras.backend.tensorflow_backend as K

from eva.layers.gated_cnn import GatedCNN, GatedCNNs
from eva.layers.masked_convolution2d import MaskedConvolution2D

def GatedPixelCNN(input_shape, filters, blocks, latent=0, build=True):
    width, height, channels = input_shape

    # TODO: Make it scalable to any amount of palettes.
    input_img = Input(shape=input_shape, name=str(channels)+'_channels_256_palette')

    latent_vector = None
    if latent != 0:
        latent_vector = Input(shape=(10,), name='latent_vector')

    model = GatedCNNs(filters, blocks, latent_vector)(*GatedCNN(filters, latent_vector)(input_img))

    for _ in range(2):
        model = Convolution2D(filters, 1, 1, border_mode='valid')(model)
        model = PReLU()(model)

    if channels == 1:
        outs = Convolution2D(1, 1, 1, activation='sigmoid', border_mode='valid')(model)
    else:
        out = Convolution2D(256*channels, 1, 1, border_mode='valid', name='channels_mult_palette')(model)
        out = Reshape((input_shape[0], input_shape[1], 256, input_shape[2]), name='palette_channels')(out)

        outs = [None] * channels
        for i in range(channels):
            outs[i] = Lambda(lambda x: x[:, :, :, :, 0], name='channel'+str(i)+'_extract')(out)
            outs[i] = Reshape((input_shape[0] * input_shape[1], 256), name='hw_palette'+str(i))(outs[i])
            outs[i] = Activation('softmax', name='channel'+str(i))(outs[i])

    if build:
        model = Model(input=[input_img, latent_vector] if latent != 0 else input_img, output=outs)
        model.compile(optimizer=Nadam(), loss='binary_crossentropy' if channels == 1 else 'sparse_categorical_crossentropy')

    return model
