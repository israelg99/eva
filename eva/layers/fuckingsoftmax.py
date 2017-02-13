from keras.engine import Layer
import tensorflow as tf


class FuckingSoftmax(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, mask=None):
        return tf.nn.softmax(x)
