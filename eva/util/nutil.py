import numpy as np


def to_rgb(pixels):
    return np.repeat(pixels, 3 if pixels.shape[2] == 1 else 1, 2)

def binarize(arr, generate=np.random.uniform):
    return (generate(size=arr.shape) < arr).astype('i')
