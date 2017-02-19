#%% Setup.
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
from keras.callbacks import TensorBoard, ModelCheckpoint

from eva.models.pixelcnn import PixelCNN
from eva.models.gated_pixelcnn import GatedPixelCNN

from eva.util.mutil import clean_data

#%% Data.
data, labels = clean_data(cifar10.load_data(), rgb=True, latent=True)

#%% Model.
# model = PixelCNN(data.shape[1:], 126, 1)
model = GatedPixelCNN(data.shape[1:], 126, 1, 1)

#%% Train.
model.fit([data, labels]
          [(np.expand_dims(data[:, :, :, 0].reshape(data.shape[0], data.shape[1]*data.shape[2]), -1)*255).astype(int),
           (np.expand_dims(data[:, :, :, 1].reshape(data.shape[0], data.shape[1]*data.shape[2]), -1)*255).astype(int),
           (np.expand_dims(data[:, :, :, 2].reshape(data.shape[0], data.shape[1]*data.shape[2]), -1)*255).astype(int)],
          batch_size=32, nb_epoch=200,
          verbose=1, callbacks=[TensorBoard(), ModelCheckpoint('model.h5', save_weights_only=True)]) # Only weights because Keras is a bitch.
