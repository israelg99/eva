import numpy as np


class Nutil(object):
    @staticmethod
    def to_rgb(pixels):
        return np.repeat(pixels, 3 if pixels.shape[2] == 1 else 1, 2)
