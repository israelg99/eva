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
from keras.callbacks import TensorBoard, ModelCheckpoint

from eva.models.pixelcnn import PixelCNN
from eva.util.nutil import quantisize

#%% Arguments.
batch_size = 32
nb_epoch = 200

data_augmentation = True

#%% Data.
(train, _), (test, _) = cifar10.load_data()
data = np.concatenate((train, test), axis=0)
data = data.astype('float32')
data /= 255

#%% Model.
model = PixelCNN(data.shape[1:], 128, 12)

model.summary()

plot(model)

#%% Train.
model.fit({'input_map': data},
          {'red': (np.expand_dims(data[:, :, :, 0].reshape(data.shape[0], data.shape[1]*data.shape[2]), -1)*255).astype(int),
           'green': (np.expand_dims(data[:, :, :, 1].reshape(data.shape[0], data.shape[1]*data.shape[2]), -1)*255).astype(int),
           'blue': (np.expand_dims(data[:, :, :, 2].reshape(data.shape[0], data.shape[1]*data.shape[2]), -1)*255).astype(int)},
          batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, callbacks=[TensorBoard(), ModelCheckpoint('model.h5')])
