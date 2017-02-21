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

#%% Args.
# Input.
TYPE = 'mnist_rgb'

# Model.
FILTERS = 126
BLOCKS = 1

# Generation.
BATCH = 10
LATENT = 4
DETERMINISTIC = False

# Parse model.
INPUT = INPUTS[TYPE]
M = GatedPixelCNN(INPUT, FILTERS, BLOCKS, None if LATENT is None else 1)
M.load_weights('model.h5')

#%% Choice (Probabilistic).
batch = generate(M, LATENT, BATCH, deterministic=DETERMINISTIC)
for pic in tqdm(batch):
    plt.imshow(pic, interpolation='nearest')
    plt.show(block=True)
