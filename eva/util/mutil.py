import numpy as np

from eva.util.nutil import binarize

def generate(model):
    pixels = np.zeros(shape=model.input_shape[1:])
    rows, cols, channels = pixels.shape
    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):
                pixels[row, col, channel] = np.argmax(model.predict_on_batch(pixels[np.newaxis])[channel][0][row*cols+col])

    return pixels

def infer(model):
    # TODO: Improve generation.
    pixels = np.zeros(shape=model.input_shape[1:])
    pixels[0,0,0] = np.random.randint(256)
    rows, cols, channels = pixels.shape
    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):
                if row == 0 and col == 0 and channel == 0:
                    continue
                pixels[row, col, channel] = np.argmax(model.predict_on_batch(pixels[np.newaxis])[channel][0][row*cols+col])

    return pixels

