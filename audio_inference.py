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

LOAD = True

#%% Data Config.
# SAMPLE_RATE = 4000
# FRAGMENT_LENGTH = 1 + compute_receptive_field(SAMPLE_RATE, DEPTH, STACKS)[0]
# FRAGMENT_STRIDE = FRAGMENT_LENGTH//10
# BINS = 256
# LEARN_ALL = True
# USE_ULAW = True
# TEST_FACTOR = 0.01
# SHUFFLE = True
# BATCH_SIZE = 8

#%% Train Config.
EPOCHS = 200

#%% Model.
INPUT = (FRAGMENT_LENGTH, BINS)
ARGS = (INPUT, FILTERS, DEPTH, STACKS)

M = MODEL(*ARGS)
if LOAD:
    M.load_weights('model.h5')

M.summary()

plot(M)

# Sample.
DIR = './samples'
BATCH_SIZE = 8
SECONDS = 5
SAMPLE_RATE = 4000
BINS = 256
FRAGMENT_LENGTH = 1+compute_receptive_field(SAMPLE_RATE, DEPTH, STACKS)[0]

def make_sample_stream(desired_sample_rate, sample_filename):
    sample_file = wave.open(sample_filename, mode='w')
    sample_file.setnchannels(1)
    sample_file.setframerate(desired_sample_rate)
    sample_file.setsampwidth(1)
    return sample_file

samples = np.zeros(shape=(BATCH_SIZE, SAMPLE_RATE*SECONDS//FRAGMENT_LENGTH*FRAGMENT_LENGTH, BINS))
for i in tqdm(range(samples.shape[1])):
    if i//FRAGMENT_LENGTH == 0:
        pr = M.predict_on_batch(samples[:, :FRAGMENT_LENGTH])[:, i]
    else:
        pr = M.predict_on_batch(samples[:, i-FRAGMENT_LENGTH+1:i+1])[:, -1]
    # samples[:, i] = np.array([np.random.choice(256, p=p) for p in pr[:, i]])
    b = np.zeros(shape=(BATCH_SIZE, BINS))
    b[:, np.array([np.random.choice(256, p=p) for p in pr])] = 1
    samples[:, i] = b

# samples = np.zeros(shape=(BATCH_SIZE, SAMPLE_RATE*SECONDS//FRAGMENT_LENGTH*FRAGMENT_LENGTH, BINS))

# M.predict_on_batch(samples[:, :FRAGMENT_LENGTH])[:, 0]

# M.input_shape