#%% Setup.
print('Importing.')
import signal
import sys

import numpy as np
import scipy.io.wavfile

from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint

from tqdm import tqdm

from eva.models.wavenet import Wavenet, compute_receptive_field

from eva.util.mutil import sparse_labels

#%% Generation Config.
print('Preparing generation.')
BATCH_SIZE = 15
LENGTH = 356415 // BATCH_SIZE

#%% Model Config.
print('Preparing the model.')
MODEL = Wavenet
FILTERS = 32
DEPTH = 7
STACKS = 4
RATE = 4000
BINS = 256

#%% Model.
print('Loading the model.')
INPUT = (LENGTH, BINS)
ARGS = (INPUT, FILTERS, DEPTH, STACKS)

M = MODEL(*ARGS)
M.load_weights('model.h5')

M.summary()

plot(M)

#%% Generate.
def save():
    print('Saving.')
    np.save('samples.npy', samples)
    np.save(type(M).__name__ + '_audio.npy', audio)

    for i in tqdm(range(BATCH_SIZE)):
        scipy.io.wavfile.write('audio' + str(i) + '.wav', RATE, audio[i])


def save_gracefully(signal, frame):
    save()
    sys.exit(0)

signal.signal(signal.SIGINT, save_gracefully)

print('Generating.')
samples = np.zeros(shape=(BATCH_SIZE, LENGTH, BINS))
audio = np.zeros(shape=(BATCH_SIZE, LENGTH))
for s in tqdm(range(BATCH_SIZE)):
    for i in tqdm(range(LENGTH)):
        samples[s, i] = M.predict_on_batch(samples[s][np.newaxis])[:, i]
        samples[s,i,np.argmax(samples[s,i])] += 1-np.sum(samples[s, i])
        audio[s, i] = np.random.choice(256, p=samples[s, i])
