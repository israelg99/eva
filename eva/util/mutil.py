import numpy as np

def generate(model, latent=None, batch=1, deterministic=False):
    get_color = lambda model, pixels, cols, row, col, channel, latent=None: model.predict_on_batch(pixels if latent is None else [pixels, latent])[channel][:, row*cols+col]
    normalize = lambda pixel: pixel/255

    shape = model.input_shape
    pixels = np.zeros(shape=(batch,) + (shape if isinstance(shape, tuple) else shape[0])[1:])
    latent_vec = None if latent is None else np.expand_dims(np.ones(batch).astype(int)*latent, -1)
    batch, rows, cols, channels = pixels.shape
    for row in range(rows):
        for col in range(cols):
            for channel in range(channels):
                ps = get_color(model, pixels, cols, row, col, channel, latent_vec)
                if deterministic:
                    pixels[:, row, col, channel] = normalize(np.array([np.argmax(p) for p in ps]))
                    continue
                pixels[:, row, col, channel] = normalize(np.array([np.random.choice(256, p=p) for p in ps]))
    return pixels
