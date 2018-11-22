from keras.models import Sequential, Model
from keras.layers import Deconv2D, Conv2D, Lambda, MaxPooling2D, UpSampling2D, TimeDistributed, Flatten, LeakyReLU, Dropout, concatenate, Input, Conv1D
from keras.initializers import Orthogonal
from keras.regularizers import l2, L1L2, l1
import keras.backend as K
import numpy as np

import tensorflow as tf

from loaders.custom import l4_loss, maximal_error

class BANDED(object):
    @staticmethod
    def build_band(input, freq_low, freq_hi, filters=2):
        size = freq_hi - freq_low
        L = Lambda(lambda x: x[:, :, freq_low:freq_hi])(input)
        L = Lambda(lambda x: K.expand_dims(x, -1))(L)
        L = Conv2D(1, (9, 1))(L) # padding - 8
        L = LeakyReLU(0.05)(Conv2D(filters, (1, size))(L))
        L = LeakyReLU(0.05)(Conv2D(filters, (9, 1))(L)) # padding - 8
        L = TimeDistributed(Flatten())(L)
        return L

    @staticmethod
    def build_deconv_band(input, index, freq_low, freq_hi, filters=2):
        size = freq_hi - freq_low
        L = Lambda(lambda x: x[:, :, index:index+1, :])(input)
        L = LeakyReLU(0.05)(Deconv2D(filters, (9, 1))(L)) # padding - 8
        L = LeakyReLU(0.05)(Deconv2D(filters, (1, size))(L))
        L = Deconv2D(filters, (9, 1))(L) # padding - 8
        L = Conv2D(1, 1)(L)
        L = TimeDistributed(Flatten())(L)
        L = Lambda(lambda x: tf.pad(x, [[0,0], [0,0], [freq_low, 129 - freq_hi]]))(L)
        return L

    @staticmethod
    def build():
        # this model is a smaller version of tight2 while maintaining layer archotecture - we try to make it a true DAE, compressing to 64 features
        # update: it's no longer the same...
        regularizing_value = 1e-4  # 3e-3
        Inp = Input((None, 129))
        bands = []
        for i in range((129 - 5) // 2): # 62 filter bands
            low = 2 * i
            hi = 2 * i + 5
            layer = BANDED.build_band(Inp, low, hi, 4)
            bands.append(layer)
        L = concatenate(bands)
        L = LeakyReLU(0.05)(Conv1D(62 * 2, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l1(regularizing_value))(L))
        hidden = LeakyReLU(0.05)(Conv1D(64, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l1(regularizing_value))(L))
        L = LeakyReLU(0.05)(Conv1D(62 * 2, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l1(regularizing_value))(hidden))
        L = LeakyReLU(0.05)(Conv1D(62 * 4, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear')(L))
        L = Lambda(lambda x: K.map_fn(lambda y: K.reshape(y, [-1, 62, 4]), x))(L)
        bands = []
        for i in range((129 - 5) // 2): # 62 filter bands
            low = 2 * i
            hi = 2 * i + 5
            layer = BANDED.build_deconv_band(L, i, low, hi, 4)
            bands.append(layer)
        low = Lambda(lambda layers: K.sum(K.stack(layers, -1), -1))(bands)
        print(low)
        L = Lambda(lambda x: K.expand_dims(x, -1))(hidden)
        L = LeakyReLU(0.05)(Deconv2D(4, [17, 2], activation='linear')(L))
        L = Conv2D(1, 1, activation='linear')(L)
        L = TimeDistributed(Flatten())(L)
        hi = Conv1D(128, 1, activation='linear')(L)
        final = concatenate([low, hi])
        model = Model(Inp, final)
        model.compile('adam', 'mae', metrics=[maximal_error, 'mse', l4_loss])
        model.summary()
        return model

