#%%
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

import keras

from eva.layers.masked_convolution2d import MaskedConvolution2D
from eva.util.mutil import generate

count = 10

model = keras.models.load_model('eva/examples/model.h5', custom_objects={'MaskedConvolution2D':MaskedConvolution2D})

#%%
for image in [generate(model) for _ in range(count)]:
    plt.imshow(image, interpolation='nearest')
    plt.show()
