from keras.models import Model
from keras.layers import Input, Convolution2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam

from eva.layers.gated_cnn import GatedCNN, GatedCNNs
from eva.layers.out_channels import OutChannels
from eva.layers.masked_convolution2d import MaskedConvolution2D

def GatedPixelCNN(input_shape, filters, depth, latent=None, build=True):
    height, width, channels = input_shape
    palette = 256 # TODO: Make it scalable to any amount of palette.

    input_img = Input(shape=input_shape, name=str(channels)+'_channels_'+str(palette)+'_palette')

    latent_vector = None
    if latent is not None:
        latent_vector = Input(shape=(latent,), name='latent_vector')

    model = GatedCNNs(filters, depth, latent_vector)(*GatedCNN(filters, latent_vector)(input_img))

    for _ in range(2):
        model = Convolution2D(filters, 1, 1, border_mode='valid')(model)
        model = PReLU()(model)

    outs = OutChannels(*input_shape, masked=False, palette=palette)(model)

    if build:
        model = Model(input=[input_img, latent_vector] if latent is not None else input_img, output=outs)
        model.compile(optimizer=Nadam(), loss='binary_crossentropy' if channels == 1 else 'sparse_categorical_crossentropy')

    return model
