#%% Setup.
import numpy as np

from keras.datasets import cifar10, mnist
from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint

from eva.models.pixelcnn import PixelCNN
from eva.models.gated_pixelcnn import GatedPixelCNN

from eva.util.mutil import clean_data

#%% Data.
DATASET = cifar10

DATA, LABELS = clean_data(DATASET.load_data(), rgb=True, latent=True)

#%% Model.
MODEL = GatedPixelCNN
FILTERS = 126
BLOCKS = 1

M = MODEL(DATA.shape[1:], FILTERS, BLOCKS, None if LABELS is None else 1)

M.summary()

plot(M)

#%% Train.
M.fit([DATA, LABELS]
          [(np.expand_dims(DATA[:, :, :, 0].reshape(DATA.shape[0], DATA.shape[1]*DATA.shape[2]), -1)*255).astype(int),
           (np.expand_dims(DATA[:, :, :, 1].reshape(DATA.shape[0], DATA.shape[1]*DATA.shape[2]), -1)*255).astype(int),
           (np.expand_dims(DATA[:, :, :, 2].reshape(DATA.shape[0], DATA.shape[1]*DATA.shape[2]), -1)*255).astype(int)],
          batch_size=32, nb_epoch=200,
          verbose=1, callbacks=[TensorBoard(), ModelCheckpoint('model.h5', save_weights_only=True)]) # Only weights because Keras is a bitch.
