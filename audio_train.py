#%% Setup.
import sys

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
BATCH = 8

#%% Model Config.
MODEL = Wavenet
FILTERS = 32
DEPTH = 10
STACKS = 4
BINS = 256
LENGTH = RATE + compute_receptive_field(RATE, DEPTH, STACKS)[0]

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
padded_data = np.zeros(DATA.shape[0]+LENGTH-1)
padded_data[LENGTH-1:] = DATA

def train_gen():
    while True:
        i = np.random.randint(0, DATA.shape[0]-2, dtype=int)
        data = padded_data[i:i+LENGTH].astype(int)
        y = data[-1][np.newaxis][np.newaxis]
        x = np_utils.to_categorical(padded_data[i:i+LENGTH], 256)[np.newaxis]
        yield x, y

# Fuck theano and its recursions <3
sys.setrecursionlimit(50000)

M.fit_generator(train_gen(), samples_per_epoch=RATE*10, nb_epoch=EPOCHS, callbacks=[ModelCheckpoint('model.h5')])
