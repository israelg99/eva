#%% Setup.
import numpy as np

from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint

from tqdm import tqdm

from eva.models.wavenet import Wavenet, compute_receptive_field

from eva.util.mutil import sparse_labels

#%% Generation Config.
BATCH_SIZE = 8
SECONDS = 3
SAMPLE = 4000

#%% Model Config.
MODEL = Wavenet
FILTERS = 32
DEPTH = 7
STACKS = 4
RATE = 4000
LENGTH = 1 + compute_receptive_field(RATE, DEPTH, STACKS)[0]
BINS = 256

#%% Model.
INPUT = (LENGTH, BINS)
ARGS = (INPUT, FILTERS, DEPTH, STACKS)

M = MODEL(*ARGS)
M.load_weights('model.h5')

M.summary()

plot(M)

#%% Sample.

# TRAINING: 355656
UNITS = SAMPLE*SECONDS//LENGTH*LENGTH

samples = np.zeros(shape=(BATCH_SIZE, UNITS, BINS))
for i in tqdm(range(UNITS)):
    if i//LENGTH == 0:
        pr = M.predict_on_batch(samples[:, :LENGTH])[:, i]
    else:
        pr = M.predict_on_batch(samples[:, i-LENGTH+1:i+1])[:, -1]
    # samples[:, i] = np.array([np.random.choice(256, p=p) for p in pr[:, i]])
    b = np.zeros(shape=(BATCH_SIZE, BINS))
    b[:, np.array([np.random.choice(256, p=p) for p in pr])] = 1
    samples[:, i] = b
