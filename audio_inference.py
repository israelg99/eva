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
from eva.util.autil import to_pcm8

#%% Generation Config.
print('Preparing generation.')
BATCH_SIZE = 1
UNITS = 356415

#%% Model Config.
print('Preparing the model.')
MODEL = Wavenet
FILTERS = 32
DEPTH = 7
STACKS = 4
RATE = 4000
BINS = 256
LENGTH = UNITS // 15

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
    np.save('audio.npy', audio)

    pcm8_64bit_wide = to_pcm8(audio)

    for i in tqdm(range(BATCH_SIZE)):
        scipy.io.wavfile.write('audio{}.wav'.format(i), RATE, pcm8_64bit_wide[i])

def save_gracefully(signal, frame):
    save()
    sys.exit(0)

signal.signal(signal.SIGINT, save_gracefully)

print('Generating.')
samples = np.zeros(shape=(BATCH_SIZE, UNITS, BINS))
audio = np.zeros(shape=(BATCH_SIZE, UNITS))
for i in tqdm(range(LENGTH)):
    samples[:, i] = M.predict(samples[:, :LENGTH], batch_size=1)[:, i]
    samples[:,i,np.argmax(samples[:,i], axis=-1)] += 1-np.sum(samples[:, i], axis=-1)
    audio[:, i] = np.array([np.random.choice(256, p=p) for p in samples[:, i]])

save()
