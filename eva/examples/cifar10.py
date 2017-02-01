#%% Imports.
import numpy as np

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Nadam
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras import backend as K
from keras.callbacks import TensorBoard

from eva.models.pixelcnn import PixelCNN
from eva.util.nutil import quantisize

#%% Arguments.
batch_size = 32
nb_epoch = 200

data_augmentation = True

#%% Data.
(train, _), (test, _) = cifar10.load_data()
features = np.concatenate((train, test), axis=0)[:1000]

# TODO: Make is scalable to any amount of channels.
# Such as: to_softmax(channel) for channel in data.shape[3].

# TODO: SPARSE IT!!!!!

RGB = (*features.shape[:3], 256)
RGB_t = (features.shape[0], features.shape[1]*features.shape[2], 256)

R = np.zeros(RGB, dtype=features.dtype)
G = np.zeros(RGB, dtype=features.dtype)
B = np.zeros(RGB, dtype=features.dtype)

for b in range(features.shape[0]):
    for c in range(features.shape[3]):
        for y in range(features.shape[1]):
            for x in range(features.shape[2]):
                indx = (b, y, x, features[b,y,x,c])
                if c == 0:
                    R[indx] = 1
                elif c == 1:
                    G[indx] = 1
                else:
                    B[indx] = 1

R = R.reshape(RGB_t)
G = G.reshape(RGB_t)
B = B.reshape(RGB_t)

#%% Model.
model = PixelCNN(features.shape[1:], 128//features.shape[3]*features.shape[3], 12)

model.summary()

plot(model)

#%% Train.
model.fit({'input_map': features},
          {'red':       R,
           'green':     G,
           'blue':      B},
          batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, callbacks=[TensorBoard()])

#%% Save model.
model.save('pixelcnn.h5')
