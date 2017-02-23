import numpy as np

def to_pcm8(data):
    return np.round(data).astype('uint8')