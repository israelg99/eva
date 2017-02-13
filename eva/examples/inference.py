#%% Import, define constants and load model.
import numpy as np

from matplotlib import pyplot as plt
from scipy.ndimage import zoom

import keras
import keras.backend as K

from eva.layers.masked_convolution2d import MaskedConvolution2D

from eva.util.mutil import generate

COUNT = 10

M = keras.models.load_model('eva/examples/model.h5', custom_objects={'MaskedConvolution2D':MaskedConvolution2D})

#%% Mask A.
a = M.get_layer(index=1)

aw = K.eval(a.W)
am = K.eval(a.mask)

print('Mask A')
print(am[3,3])
print((am*aw)[3,3])

#%% Mask B.
b = M.get_layer(index=-8)

bw = K.eval(b.W)
bm = K.eval(b.mask)

print('Mask B')
print(bm[0,0])
print(bm*bw[0,0])

#%% Full Prediction Distribution of First Channel.
print(M.predict_on_batch(np.zeros(shape=M.input_shape[1:])[np.newaxis])[0][0][0]/255)

#%% Channel Prediction Distribution.
print('RED')
for _ in range(10):
    print(np.random.choice(256, p=M.predict_on_batch(np.zeros(shape=M.input_shape[1:])[np.newaxis])[0][0][0])/255)

print('GREEN')
for _ in range(10):
    print(np.random.choice(256, p=M.predict_on_batch(np.zeros(shape=M.input_shape[1:])[np.newaxis])[1][0][0])/255)

print('BLUE')
for _ in range(10):
    print(np.random.choice(256, p=M.predict_on_batch(np.zeros(shape=M.input_shape[1:])[np.newaxis])[2][0][0])/255)

#%% Data Test.
r=(np.expand_dims(data[:, :, :, 0].reshape(data.shape[0], data.shape[1]*data.shape[2]), -1)*255).astype(int)
g=(np.expand_dims(data[:, :, :, 1].reshape(data.shape[0], data.shape[1]*data.shape[2]), -1)*255).astype(int)
b=(np.expand_dims(data[:, :, :, 2].reshape(data.shape[0], data.shape[1]*data.shape[2]), -1)*255).astype(int)

plt.imshow(np.concatenate((r,g,b), axis=2)[1].reshape(28,28,3)/255, interpolation='nearest')
plt.show()
plt.imshow(data[1], interpolation='nearest')
plt.show()

#%% Training Pass.
pixels = np.zeros(shape=M.input_shape[1:])

pred = M.predict_on_batch(data[1][np.newaxis])
rows, cols, channels = pixels.shape
for row in range(rows):
    for col in range(cols):
        for channel in range(channels):
            pixels[row, col, channel] = np.argmax(pred[channel][0][row*cols+col])/255

plt.imshow(pixels, interpolation='nearest')
plt.show()

plt.imshow(data[1], interpolation='nearest')
plt.show()

#%% Choice (Probabilistic).
plt.imshow(generate(M, deterministic=False), interpolation='nearest')
plt.show()

#%% ArgMax (Deterministic)
plt.imshow(generate(M, deterministic=True), interpolation='nearest')
plt.show()

#%% Default Generation.
for image in [generate(M) for _ in range(COUNT)]:
    plt.imshow(image, interpolation='nearest')
    plt.show()
