import numpy as np

from tqdm import tqdm

def generate(model, latent=None, batch=1, deterministic=False):
    get_color = lambda model, pixels, cols, row, col, channel, latent=None: model.predict_on_batch(pixels if latent is None else [pixels, latent])[channel][:, row*cols+col]
    normalize = lambda pixel: pixel/255

    shape = model.input_shape
    pixels = np.zeros(shape=(batch,) + (shape if isinstance(shape, tuple) else shape[0])[1:])
    latent_vec = None if latent is None else np.expand_dims(np.ones(batch).astype(int)*latent, -1)
    batch, rows, cols, channels = pixels.shape
    for row in tqdm(range(rows)):
        for col in tqdm(range(cols)):
            for channel in tqdm(range(channels)):
                ps = get_color(model, pixels, cols, row, col, channel, latent_vec)
                if deterministic:
                    pixels[:, row, col, channel] = normalize(np.array([np.argmax(p) for p in ps]))
                    continue
                pixels[:, row, col, channel] = normalize(np.array([np.random.choice(256, p=p) for p in ps]))
    return pixels

def clean_data(data, rgb=True, latent=True):
    (train, train_l), (test, test_l) = data
    data = np.concatenate((train, test), axis=0)
    data = data.astype('float32')
    data /= 255

    assert len(data.shape) == 3 or len(data.shape) == 4

    if len(data.shape) == 3:
        data = np.expand_dims(data, -1)

    if rgb:
        data = np.repeat(data, 1 if data.shape[3] == 3 else 3, 3)

    labels = None
    if latent:
        labels = np.concatenate((train_l, test_l), axis=0)

        if len(labels.shape) == 1:
            labels = labels[np.newaxis].T

        assert len(labels.shape) == 2
        assert labels.shape[0] == data.shape[0]

    return data, labels

def sparse_labels(labels):
    return np.expand_dims(np.argmax(labels, -1), -1)

def sparse_labels_generator(generator):
    while True:
        inputs, labels = next(generator)
        yield (inputs, sparse_labels(labels))