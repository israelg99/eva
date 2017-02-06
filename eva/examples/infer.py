#%%
from matplotlib import pyplot as plt

import keras

from eva.layers.masked_convolution2d import MaskedConvolution2D
from eva.layers.color_extract import ColorExtract

from eva.util.mutil import infer, generate

count = 1

model = keras.models.load_model('eva/examples/model.h5', custom_objects={'MaskedConvolution2D':MaskedConvolution2D})

model.get_layer(index=-7).get_weights()[0].shape

#%%
import numpy as np #RRR
pixels = np.zeros(shape=model.input_shape[1:])
p = model.predict_on_batch(pixels[np.newaxis])
t = np.zeros(256)

for c in p:
    for i in c[0]:
        t[np.argmax(i)] += 1

print(t)

#%%
for image in [generate(model) for _ in range(count)]:
    plt.imshow(image, interpolation='nearest')
    plt.show()

for image in [infer(model) for _ in range(count)]:
    plt.imshow(image, interpolation='nearest')
    plt.show()
