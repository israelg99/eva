#%%
from matplotlib import pyplot as plt

import keras

from eva.layers.masked_convolution2d import MaskedConvolution2D

from eva.util.mutil import Mutil
from eva.util.nutil import Nutil

count = 1
model = keras.models.load_model('eva/examples/pixelcnn.h5', custom_objects={'MaskedConvolution2D':MaskedConvolution2D})

for image in [Nutil.to_rgb(Mutil.infer(model)) for _ in range(count)]:
    plt.imshow(image, interpolation='nearest')
    plt.show()
