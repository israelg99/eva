from keras.layers import Lambda, ZeroPadding2D
import keras.backend as K


def ShiftDown(model):
    shape = K.int_shape(model)[1]
    model = ZeroPadding2D(padding=(1,0,0,0))(model)
    model = Lambda(lambda x: x[:,:shape,:,:])(model)
    return model
