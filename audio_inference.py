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
UNITS = 356415

#%% Model Config.
print('Preparing the model.')
MODEL = Wavenet
FILTERS = 32
DEPTH = 8
STACKS = 4
BINS = 256
SAMPLE = 4000
LENGTH = 1 + compute_receptive_field(SAMPLE, DEPTH, STACKS)[0]

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
        scipy.io.wavfile.write('audio{}.wav'.format(i), SAMPLE, pcm8_64bit_wide[i])

def save_gracefully(signal, frame):
    save()
    sys.exit(0)

signal.signal(signal.SIGINT, save_gracefully)

BATCH_SIZE = 1

samples = np.zeros(shape=(BATCH_SIZE, UNITS, BINS))
audio = np.zeros(shape=(BATCH_SIZE, UNITS))
for i in tqdm(range(UNITS+LENGTH-1)):
    if i >= UNITS-1:
        break
    x = i+LENGTH-1
    samples[:, x] = M.predict(samples[:, i:i+LENGTH], batch_size=1)[0]
    samples[:,x,np.argmax(samples[:,x], axis=-1)] += 1-np.sum(samples[:, x], axis=-1)
    audio[:, x] = np.array([np.random.choice(256, p=p) for p in samples[:, x]])
    if i>0 and ((i-LENGTH+1) % (SAMPLE//2) == 0):
        print(str((i-LENGTH+1)/(SAMPLE//2)) + " Seconds generated!")

save()