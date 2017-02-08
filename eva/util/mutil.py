import numpy as np

def generate(model):
    pixels = np.zeros(shape=model.input_shape[1:])
    rows, cols, channels = pixels.shape
    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):
                pixels[row, col, channel] = np.random.choice(256, p=model.predict_on_batch(pixels[np.newaxis])[channel][row*cols+col])
                # pixels[row, col, channel] = np.random.choice(256, p=model.predict_on_batch(pixels[np.newaxis])[channel][0][row*cols+col])

    return pixels
