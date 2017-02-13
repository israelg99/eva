import numpy as np

def generate(model, deterministic=False):
    get_color = lambda model, pixels, row, col, channel: model.predict_on_batch(pixels[np.newaxis])[channel][0][row*cols+col]
    to_255 = lambda color: color/255

    pixels = np.zeros(shape=model.input_shape[1:])
    rows, cols, channels = pixels.shape
    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):
                if deterministic:
                    pixels[row, col, channel] = to_255(np.argmax(get_color(model, pixels, row, col, channel)))
                    continue
                pixels[row, col, channel] = to_255(np.random.choice(256, p=get_color(model, pixels, row, col, channel)))
    return pixels
