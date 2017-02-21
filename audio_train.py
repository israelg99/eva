#%% Setup.
import numpy as np

from IPython import embed

from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint

from eva.data.audio_reader import *

from eva.models.wavenet import Wavenet, compute_receptive_field

from eva.util.mutil import clean_data, sparse_targets

#%% Model Config.
MODEL = Wavenet
FILTERS = 32
DEPTH = 7
STACKS = 4

LOAD = False

#%% Data Config.
SAMPLE_RATE = 4000
FRAGMENT_LENGTH = 1 + compute_receptive_field(SAMPLE_RATE, DEPTH, STACKS)[0]
FRAGMENT_STRIDE = FRAGMENT_LENGTH//10
BINS = 256
LEARN_ALL = True
USE_ULAW = True
TEST_FACTOR = 0.01
SHUFFLE = True
BATCH_SIZE = 8

#%% Train Config.
EPOCHS = 200

#%% Data.

# TODO: More preprocessing.
# TODO: Ensure the data is valid.
# TODO: Chop silent parts with a threshold.
generators, examples = generators_vctk('./data/vctk/wav48',
                                       SAMPLE_RATE, FRAGMENT_LENGTH, BATCH_SIZE,
                                       FRAGMENT_STRIDE, BINS, LEARN_ALL, USE_ULAW,
                                       TEST_FACTOR, SHUFFLE, SHUFFLE)

#%% Model.
INPUT = (FRAGMENT_LENGTH, BINS)
ARGS = (INPUT, FILTERS, DEPTH, STACKS)

M = MODEL(*ARGS)
if LOAD:
    M.load_weights('model.h5')

M.summary()

plot(M)

#%% Train.
M.fit_generator(sparse_targets(generators['train']), samples_per_epoch=examples['train'],
      nb_epoch=EPOCHS, verbose=1, callbacks=[TensorBoard(), ModelCheckpoint('model.h5')])

#%% Debug.
# d = next(sparse_targets(generators['train']))
# M.fit(d[0], d[1], 8, 2, 1)
