from keras.models import Model
from keras.layers import Input, Convolution2D, Activation, Flatten, Dense, Reshape, Lambda
from keras.layers.advanced_activations import PReLU
from keras.engine.topology import merge
from keras.optimizers import Nadam
import keras.backend.tensorflow_backend as K

from eva.layers.residual_block import ResidualBlockList
from eva.layers.masked_convolution2d import MaskedConvolution2D
from eva.layers.fuckingsoftmax import FuckingSoftmax

def PixelCNN(input_shape, filters, blocks, build=True):
    width, height, channels = input_shape

    input_map = Input(shape=input_shape, name='input_map')

    model = MaskedConvolution2D(filters, 7, 7, mask='A', border_mode='same')(input_map)
    model = PReLU()(model)

    model = ResidualBlockList(model, filters, blocks)

    model = MaskedConvolution2D(filters, 1, 1)(model)
    model = PReLU()(model)

    model = MaskedConvolution2D(3*256, 1, 1)(model)
    model = PReLU()(model)

    # TODO: Make it scalable to any amount of channels.

    model = Reshape((input_shape[0], input_shape[1], 256, input_shape[2]))(model)

    # TODO: Make it scalable to any amount of channels.

    red = Lambda(lambda x: x[:, :, :, :, 0])(model)
    red = Reshape((input_shape[0] * input_shape[1], 256))(red)
    red = Activation('softmax', name='red')(red)

    green = Lambda(lambda x: x[:, :, :, :, 1])(model)
    green = Reshape((input_shape[0] * input_shape[1], 256))(green)
    green = Activation('softmax', name='green')(green)

    blue = Lambda(lambda x: x[:, :, :, :, 2])(model)
    blue = Reshape((input_shape[0] * input_shape[1], 256))(blue)
    blue = Activation('softmax', name='blue')(blue)

    # TODO: Make is scalable to any amount of channels.

    if build:
        model = Model(input=input_map, output=[red, green, blue])
        model.compile(optimizer=Nadam(),
                      loss={    'red':   image_categorical_crossentropy,
                                'green': image_categorical_crossentropy,
                                'blue':  image_categorical_crossentropy})

    return model

from keras.backend.common import _EPSILON
from keras.backend.tensorflow_backend import _to_tensor, cast, flatten
import tensorflow as tf
def image_categorical_crossentropy(y_true, y_pred, from_logits=False):
    if not from_logits:
        epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = tf.log(y_pred)

    output_shape = y_pred.get_shape()
    targets = cast(flatten(y_true), 'int64')
    logits = tf.reshape(y_pred, [-1, int(output_shape[-1])])

    res = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets,
            logits=logits)

    res = tf.reshape(res, tf.shape(y_pred)[:-1])

    return tf.reduce_sum(res, -1)
