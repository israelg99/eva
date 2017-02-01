import numpy as np

from eva.util.nutil import binarize

def infer(model):
    pixels = np.random.rand(*model.input_shape[1:])
    rows, cols, channels = pixels.shape
    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):
                pixels[row, col, channel] = np.argmax(model.predict_on_batch(pixels[np.newaxis])[channel][0][row*cols+col])

    return pixels
