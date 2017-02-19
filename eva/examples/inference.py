#%% Setup.
import numpy as np

from matplotlib import pyplot as plt
from scipy.ndimage import zoom

import keras
from keras.utils import np_utils
import keras.backend as K

from eva.models.gated_pixelcnn import GatedPixelCNN

from eva.util.mutil import generate

INPUTS = {
    'cifar10' : (32, 32, 3),
    'mnist_rgb' : (28, 28, 3),
    'mnist' : (28, 28, 1)
}

#%% Args.
# Input.
TYPE = 'mnist_rgb'

# Model.
FILTERS = 126
BLOCKS = 1
CLASSES = 10

# Generation.
BATCH = 10
LATENT = None

# Parse model.
INPUT = INPUTS[TYPE]
M = GatedPixelCNN(INPUT, FILTERS, BLOCKS, LATENT)
M.load_weights('eva/examples/model.h5')

#%% Choice (Probabilistic).
batch = generate(M, LATENT, BATCH, deterministic=False)
for pic in batch:
    plt.imshow(pic, interpolation='nearest')
    plt.show()

#%% ArgMax (Deterministic)
batch = generate(M, LATENT, BATCH, deterministic=True)
for pic in batch:
    plt.imshow(pic, interpolation='nearest')
    plt.show()

#%% Test generation.
# get_color = lambda model, pixels, cols, row, col, channel, latent=None: model.predict_on_batch(pixels if latent is None else [pixels, latent])[channel][:, row*cols+col]
# to_255 = lambda color: color

# shape = M.input_shape
# pixels = np.zeros(shape=(BATCH,) + (shape if isinstance(shape, tuple) else shape[0])[1:])
# latent_vec = None
# batch, rows, cols, channels = pixels.shape
# for row in range(rows):
#     for col in range(cols):
#         for channel in range(channels):
#             ps = get_color(M, pixels, cols, row, col, channel, latent_vec)
#             if deterministic:
#                 pixels[:, row, col, channel] = to_255(np.array([np.argmax(p) for p in ps]))
#                 continue
#             pixels[:, row, col, channel] = to_255(np.array([np.random.choice(256, p=p) for p in ps]))
# return pixels
# ps = get_color(M, pixels, 32, 0, 1, 0)
# pixels[:, 0, 1, 0] = to_255(np.array([np.random.choice(256, p=p) for p in ps]))