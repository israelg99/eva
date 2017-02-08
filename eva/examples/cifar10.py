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

features_3c = data
labels_3c = np.expand_dims(features_3c.reshape(features_3c.shape[0], features_3c.shape[1]*features_3c.shape[2], features_3c.shape[3]), -1)

features_1c = np.expand_dims(np.dot(data, [0.299, 0.587, 0.114]), -1)
labels_1c = features_1c.reshape(features_1c.shape[0], features_1c.shape[1]*features_1c.shape[2], features_1c.shape[3]).astype(int)

# TODO: Make is scalable to any amount of channels.
# Such as: to_softmax(channel) for channel in data.shape[3].

#%% Model.
model = PixelCNN(features_3c.shape[1:], 128, 12)

model.summary()

plot(model)

features_3c

#%% Train.
model.fit({'input_map': features_3c},
          {'red': np.expand_dims(features_3c[:, :, :, 0].reshape(features_3c.shape[0], features_3c.shape[1]*features_3c.shape[2]), -1),
           'green': np.expand_dims(features_3c[:, :, :, 1].reshape(features_3c.shape[0], features_3c.shape[1]*features_3c.shape[2]), -1),
           'blue': np.expand_dims(features_3c[:, :, :, 2].reshape(features_3c.shape[0], features_3c.shape[1]*features_3c.shape[2]), -1)},
          batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, callbacks=[TensorBoard(), ModelCheckpoint('model.h5')])

# model.fit(features_3c, labels_1c,
#           batch_size=batch_size, nb_epoch=nb_epoch,
#           verbose=1, callbacks=[TensorBoard(), ModelCheckpoint('model.h5')])
