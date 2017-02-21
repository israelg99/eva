#%% Setup.
print('Importing.')
import numpy as np
import scipy.io.wavfile

from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint

from tqdm import tqdm

from eva.models.wavenet import Wavenet, compute_receptive_field

from eva.util.mutil import sparse_labels

#%% Generation Config.
print('Preparing generation.')
BATCH_SIZE = 8
SECONDS = 3
SAMPLE = 4000

#%% Model Config.
print('Preparing the model.')
MODEL = Wavenet
FILTERS = 32
DEPTH = 7
STACKS = 4
RATE = 4000
LENGTH = 1 + compute_receptive_field(RATE, DEPTH, STACKS)[0]
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
print('Generating.')

# TRAINING: 355656
UNITS = SAMPLE*SECONDS//LENGTH*LENGTH

samples = np.zeros(shape=(BATCH_SIZE, UNITS, BINS))
audio = np.zeros(shape=(BATCH_SIZE, UNITS))
for i in tqdm(range(UNITS)):
    if i//LENGTH == 0:
        pr = M.predict_on_batch(samples[:, :LENGTH])[:, i]
    else:
        pr = M.predict_on_batch(samples[:, i-LENGTH+1:i+1])[:, -1]
    samples[:, i] = pr
    audio[:, i] = np.array([np.random.choice(256, p=p) for p in pr])


#%% Save.
print('Saving.')
samples = np.save(type(M).__name__ + '_samples.npy', samples)
audio = np.save(type(M).__name__ + '_audio.npy', audio)

for i in tqdm(range(BATCH_SIZE)):
    scipy.io.wavfile.write(type(M).__name__ + '_audio.wav', RATE, audio[i])