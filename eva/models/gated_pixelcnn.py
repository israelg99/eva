from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, Flatten, Dense, Reshape, Lambda
from keras.layers.advanced_activations import PReLU
from keras.engine.topology import merge
from keras.optimizers import Nadam
import keras.backend.tensorflow_backend as K

from eva.layers.gated_cnn import GatedCNN, GatedCNNs
from eva.layers.masked_convolution2d import MaskedConvolution2D

def GatedPixelCNN(input_shape, filters, blocks, latent=None, build=True):
    height, width, channels = input_shape
    palette = 256

    # TODO: Make it scalable to any amount of palettes.
    input_img = Input(shape=input_shape, name=str(channels)+'_channels_256_palette')

    latent_vector = None
    if latent is not None:
        latent_vector = Input(shape=(latent,), name='latent_vector')

    model = GatedCNNs(filters, blocks, latent_vector)(*GatedCNN(filters, latent_vector)(input_img))

    for _ in range(2):
        model = Convolution2D(filters, 1, 1, border_mode='valid')(model)
        model = PReLU()(model)

    if channels == 1:
        outs = Convolution2D(1, 1, 1, activation='sigmoid', border_mode='valid')(model)
    else:
        out = Convolution2D(palette*channels, 1, 1, border_mode='valid', name='channels_mult_palette')(model)
        out = Reshape((height, width, palette, channels), name='palette_channels')(out)

        outs = [None] * channels
        for i in range(channels):
            outs[i] = Lambda(lambda x: x[:, :, :, :, i], name='channel'+str(i)+'_extract')(out)
            outs[i] = Reshape((height * width, palette), name='hw_palette'+str(i))(outs[i])
            outs[i] = Activation('softmax', name='channel'+str(i))(outs[i])

    if build:
        model = Model(input=[input_img, latent_vector] if latent is not None else input_img, output=outs)
        model.compile(optimizer=Nadam(), loss='binary_crossentropy' if channels == 1 else 'sparse_categorical_crossentropy')

    return model
