import numpy as np


def to_rgb(pixels):
    return np.repeat(pixels, 3 if pixels.shape[2] == 1 else 1, 2)

def binarize(arr, generate=np.random.uniform):
    """
    Stochastically binarize values in [0, 1] by treating them as p-values of
    a Bernoulli distribution.
    """
    return (generate(size=arr.shape) < arr).astype('float32')
