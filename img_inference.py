#%% Setup.
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

from eva.models.gated_pixelcnn import GatedPixelCNN

from eva.util.mutil import generate

INPUTS = {
    'cifar10' : (32, 32, 3),
    'mnist_rgb' : (28, 28, 3),
    'mnist' : (28, 28, 1)
}

#%% Model.
TYPE = 'mnist_rgb'
MODEL = GatedPixelCNN
FILTERS = 126
DEPTH = 12

#%% Generation.
BATCH = 10
LATENT = 4
DETERMINISTIC = False

#%% Parse model.
ARGS = (INPUTS[TYPE], FILTERS, DEPTH)
if MODEL == GatedPixelCNN and LABELS is not None:
    ARGS += (1,)

M = MODEL(*ARGS)
M.load_weights('model.h5')


#%% Choice (Probabilistic).
batch = generate(M, LATENT, BATCH, deterministic=DETERMINISTIC)
for pic in tqdm(batch):
    plt.imshow(pic, interpolation='nearest')
    plt.show(block=True)
