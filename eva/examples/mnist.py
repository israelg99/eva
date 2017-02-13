#%% Setup.
from collections import namedtuple

import numpy as np

import keras
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

Data = namedtuple('Data', 'x y')

nb_classes = 10
img_rows, img_cols = 28, 28

nb_filters = 128
blocks = 12

batch_size = 128
nb_epoch = 40

def clean_data(x, y, rows, cols):
    if K.image_dim_ordering() == 'th':
        x = x.reshape(x.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        x = x.reshape(x.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)

    x = x.astype('float32') / 255

    y = np_utils.to_categorical(y, nb_classes)

    print('X shape:', x.shape)
    print(x.shape[0], 'samples')

    return x, y

def get_data(rows, cols):
    return [Data(*clean_data(*data, rows, cols)) for data in mnist.load_data()]

train, test = get_data(img_rows, img_cols)
data = np.concatenate((train.x, test.x))

data = np.repeat(data, 3, -1)

model = PixelCNN(data.shape[1:], nb_filters, blocks)

model.summary()

plot(model)

#%% Train.
model.fit({'input_map': data},
          {'red': (np.expand_dims(data[:, :, :, 0].reshape(data.shape[0], data.shape[1]*data.shape[2]), -1)*255).astype(int),
           'green': (np.expand_dims(data[:, :, :, 1].reshape(data.shape[0], data.shape[1]*data.shape[2]), -1)*255).astype(int),
           'blue': (np.expand_dims(data[:, :, :, 2].reshape(data.shape[0], data.shape[1]*data.shape[2]), -1)*255).astype(int)},
          batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, callbacks=[TensorBoard(), ModelCheckpoint('model.h5')])
