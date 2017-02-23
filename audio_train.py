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
EPOCHS = 2000

#%% Model Config.
MODEL = Wavenet
FILTERS = 32
DEPTH = 8
STACKS = 4
BINS = 256
SAMPLE = 4000
LENGTH = 1 + compute_receptive_field(SAMPLE, DEPTH, STACKS)[0]

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
BATCH_SIZE = DATA.shape[0]//LENGTH
TRAIN = np_utils.to_categorical(DATA[:BATCH_SIZE*LENGTH], BINS).reshape(BATCH_SIZE, LENGTH, BINS)

M.fit(TRAIN, sparse_labels(TRAIN)[:, -1], nb_epoch=EPOCHS, batch_size=1,
      callbacks=[TensorBoard(), ModelCheckpoint('model.h5', save_weights_only=True)])