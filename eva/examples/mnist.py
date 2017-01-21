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

from eva.models.pixelcnn import PixelCNN

Data = namedtuple('Data', 'x y')

nb_classes = 10
img_rows, img_cols = 28, 28

nb_filters = 128
blocks = 4

batch_size = 128
nb_epoch = 4

def clean_data(x, y, rows, cols):
    if K.image_dim_ordering() == 'th':
        x = x.reshape(x.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        x = x.reshape(x.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)

    x = x.astype('float32') / 255

    y = np_utils.to_categorical(y, nb_classes)

    # New way
    x[np.where(x > 0)] = 1

    print('X shape:', x.shape)
    print(x.shape[0], 'samples')

    return x, y

def get_data(rows, cols):
    return [Data(*clean_data(*data, rows, cols)) for data in mnist.load_data()]

def get_input(rows, cols):
    return (1, rows, cols) if K.image_dim_ordering() == 'th' else (rows, cols, 1)

train, test = get_data(img_rows, img_cols)
input_shape = get_input(img_rows, img_cols)
input_dims = np.prod(input_shape)

model = PixelCNN(input_shape, nb_filters, blocks)

model.summary()

plot(model)

#%% Train.
model.fit(train.x, train.x, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(test.x, test.x))

score = model.evaluate(test.x, test.x, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#%% Save model.
model.save('pixelcnn.h5')
