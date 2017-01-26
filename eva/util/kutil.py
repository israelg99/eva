import keras.backend as K

def get_input(rows, cols, channels):
    return (channels, rows, cols) if K.image_dim_ordering() == 'th' else (rows, cols, channels)
