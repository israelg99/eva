#%%
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

import keras

from eva.layers.masked_convolution2d import MaskedConvolution2D
from eva.util.mutil import generate


count = 10

model = keras.models.load_model('eva/examples/model.h5', custom_objects={'MaskedConvolution2D':MaskedConvolution2D})

import numpy as np
pixels = np.zeros(shape=model.input_shape[1:])
rows, cols, channels = pixels.shape
for row in range(rows):
    for col in range(cols):
        pixels[row, col, 0] = np.random.choice(256, p=model.predict_on_batch(pixels[np.newaxis])[0][row*cols+col])
pixels = pixels[:, :, 0]

plt.imshow(zoom(pixels, 3, order=0), interpolation='lanczos')
plt.show()

#%%
for image in [generate(model) for _ in range(count)]:
    plt.imshow(image, interpolation='nearest')
    plt.show()
