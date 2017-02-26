import numpy as np
from keras.models import Model
from keras.layers import Input, AtrousConvolution1D

inp = Input((100, 1))
M = AtrousConvolution1D(1, 2, atrous_rate=25, border_mode='same')(inp)
M = Model(inp, M)
M.compile('sgd', 'mse')

M.train_on_batch(np.random.rand(1, 100, 1), np.random.rand(1, 100, 1))