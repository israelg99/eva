from keras.models import Model
from keras.layers import Input
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam

from eva.layers.residual_block import ResidualBlockList
from eva.layers.out_channels import OutChannels
from eva.layers.masked_convolution2d import MaskedConvolution2D

def PixelCNN(input_shape, filters, depth, build=True):
    height, width, channels = input_shape
    palette = 256 # TODO: Make it scalable to any amount of palette.

    input_img = Input(shape=input_shape, name=str(channels)+'_channels_'+str(palette)+'_palette')

    model = MaskedConvolution2D(filters, 7, 7, mask='A', border_mode='same', name='masked2d_A')(input_img)

    model = ResidualBlockList(filters, depth)(model)
    model = PReLU()(model)

    for _ in range(2):
        model = MaskedConvolution2D(filters, 1, 1, border_mode='valid')(model)
        model = PReLU()(model)

    outs = OutChannels(*input_shape, masked=True, palette=palette)(model)

    if build:
        model = Model(input=input_img, output=outs)
        model.compile(optimizer=Nadam(), loss='binary_crossentropy' if channels == 1 else 'sparse_categorical_crossentropy')

    return model
