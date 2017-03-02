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

#%% Data.
RATE, DATA = scipy.io.wavfile.read('./data/undertale/undertale_001_once_upon_a_time.comp.wav')

#%% Generation Config.
print('Preparing generation.')
UNITS = 356415

#%% Model Config.
print('Preparing the model.')
MODEL = Wavenet
FILTERS = 32
DEPTH = 10
STACKS = 5
BINS = 256
LAST = RATE
LENGTH = LAST + compute_receptive_field(RATE, DEPTH, STACKS)[0]


#%% Model.
print('Loading the model.')
INPUT = (LENGTH, BINS)
ARGS = (INPUT, FILTERS, DEPTH, STACKS)

M = MODEL(*ARGS)
M.load_weights('model.h5')

M.summary()

plot(M)

#%% Generate.
print('Generating.')
def save():
    print('Saving.')
    np.save('samples.npy', samples)
    np.save('audio.npy', audio)

    pcm8_64bit_wide = to_pcm8(audio)

    for i in tqdm(range(audio.shape[0])):
        scipy.io.wavfile.write('audio{}.wav'.format(i), RATE, pcm8_64bit_wide[i])

def save_gracefully(signal, frame):
    save()
    sys.exit(0)

signal.signal(signal.SIGINT, save_gracefully)

BATCH_SIZE = 1

samples = np.zeros(shape=(1, UNITS, BINS))
samples[0, list(range(LENGTH-1)), DATA[:LENGTH-1].astype(int)]
samples = np.repeat(samples, BATCH_SIZE, axis=0)

audio = np.zeros(shape=(1, UNITS))
audio[0, :LENGTH-1] = DATA[:LENGTH-1]
audio = np.repeat(audio, BATCH_SIZE, axis=0)

for i in tqdm(range(UNITS+LENGTH-1)):
    if i >= UNITS-1:
        break
    x = i+LENGTH-1
    samples[:, x] = M.predict(samples[:, i:x+1], batch_size=1)[:, -1]
    samples[:,x,np.argmax(samples[:,x], axis=-1)] += 1-np.sum(samples[:, x], axis=-1)
    audio[:, x] = np.array([np.random.choice(256, p=p) for p in samples[:, x]])
    if i % (RATE//2) == 0:
        print(str(i/(RATE)) + " Seconds generated!")
        save()

save()