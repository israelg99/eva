from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, Flatten, Dense
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam

from eva.models.residual_block import ResidualBlockList
from eva.layers.masked_convolution2d import MaskedConvolution2D

def PixelCNN(input_shape, filters, blocks, softmax=False, build=True):
    input_map = Input(shape=input_shape)

    model = MaskedConvolution2D(filters, 7, 7, mask='A', border_mode='same')(input_map)
    model = PReLU()(model) # Against paper, relu shouldn't be here.

    model = ResidualBlockList(model, filters, blocks)
    # Against paper, should be relu here.

    # Against paper, should be Convolution Mask B 1x1, which I am yet to make sense of.
    model = Convolution2D(filters, 1, 1)(model)
    model = PReLU()(model)

    # Against paper again, same as above, and also paper didn't mention solo filter but I assume it is obvious.
    model = Convolution2D(1, 1, 1)(model)

    if not softmax:
        model = Activation('sigmoid')(model)
    else:
        raise NotImplementedError()

    if build:
        # (Potentially) Against paper, loss and optimizers are different.
        model = Model(input=input_map, output=model)
        model.compile(loss='kld',
                      optimizer=Nadam(),
                      metrics=['accuracy'])

    return model
