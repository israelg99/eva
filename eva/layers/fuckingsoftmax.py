from keras.engine import Layer
from keras import backend as K

class FuckingSoftmax(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, mask=None):
        return K.softmax(x)

    def get_config(self):
        return dict(list(super().get_config().items()) + list({'fucking': True}.items()))
