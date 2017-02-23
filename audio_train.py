#%% Setup.
import numpy as np
import scipy.io.wavfile

from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import np_utils

from eva.models.wavenet import Wavenet, compute_receptive_field

from eva.util.mutil import sparse_labels

#%% Data
RATE, DATA = scipy.io.wavfile.read('./data/undertale/undertale_001_once_upon_a_time.comp.wav')

#%% Train Config.
BATCH_SIZE = 15
EPOCHS = 2000
LENGTH = DATA.shape[0] // BATCH_SIZE

#%% Model Config.
MODEL = Wavenet
FILTERS = 32
DEPTH = 7
STACKS = 4
BINS = 256

LOAD = False

#%% Model.
INPUT = (LENGTH, BINS)
ARGS = (INPUT, FILTERS, DEPTH, STACKS)

M = MODEL(*ARGS)
if LOAD:
    M.load_weights('model.h5')

M.summary()

plot(M)

#%% Train.
TRAIN = np_utils.to_categorical(DATA, BINS)
TRAIN = TRAIN.reshape(BATCH_SIZE, TRAIN.shape[0]//BATCH_SIZE, TRAIN.shape[1])

M.fit(TRAIN, sparse_labels(TRAIN), nb_epoch=EPOCHS, batch_size=1,
      callbacks=[TensorBoard(), ModelCheckpoint('model.h5', save_weights_only=True)])