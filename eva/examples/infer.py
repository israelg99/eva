#%%
from matplotlib import pyplot as plt

import keras

from eva.layers.masked_convolution2d import MaskedConvolution2D
from eva.layers.color_extract import ColorExtract

from eva.util.mutil import infer
from eva.util.nutil import to_rgb

model = keras.models.load_model('eva/examples/pixelcnn.h5', custom_objects={'MaskedConvolution2D':MaskedConvolution2D, 'ColorExtract':ColorExtract})

for image in [infer(model) for _ in range(count)]:
    plt.imshow(image, interpolation='nearest')
    plt.show()
