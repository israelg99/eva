import numpy as np
from matplotlib import pyplot as plt


class Mutil(object):
    @staticmethod
    def to_rgb_arrays(model, count):
        return np.repeat(model.predict_on_batch(np.zeros((count,) + model.input_shape[1:])), 3 if model.input_shape[3] == 1 else 1, 3)

    @staticmethod
    def display_rgb_output(model, count):
        for image in Mutil.to_rgb_arrays(model, count):
            plt.imshow(image, interpolation='nearest')
            plt.show()
