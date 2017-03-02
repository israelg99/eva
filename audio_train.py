#%% Setup.
import signal
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
STACKS = 5
BINS = 256
LAST = RATE
LENGTH = LAST + compute_receptive_field(RATE, DEPTH, STACKS)[0]

LOAD = False

#%% Model.
INPUT = (LENGTH, BINS)
ARGS = (INPUT, FILTERS, DEPTH, STACKS, LAST)

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
        i = np.random.randint(0, DATA.shape[0]-2, size=BATCH, dtype=int)

        data = np.zeros((BATCH, LENGTH))
        y = np.zeros((BATCH, LAST, 1))
        x = np.zeros((BATCH, LENGTH, BINS))
        for s in range(BATCH):
            si = i[s]
            data[s] = padded_data[si:si+LENGTH].astype(int)
            y[s] = np.expand_dims(data[s, -LAST:], -1)
            x[s, list(range(LENGTH)), data[s].astype(int)] = 1
        yield x, y

def save():
    M.save('sigint_model.h5')

def save_gracefully(signal, frame):
    save()
    sys.exit(0)

signal.signal(signal.SIGINT, save_gracefully)

# Fuck theano and its recursions <3
sys.setrecursionlimit(50000)

M.fit_generator(train_gen(), samples_per_epoch=RATE//4, nb_epoch=EPOCHS, callbacks=[ModelCheckpoint('model.h5')])
