from keras.models import Sequential, Model
from keras.layers import Deconv2D, Conv2D, Lambda, MaxPooling2D, UpSampling2D, TimeDistributed, Flatten, LeakyReLU, Dropout, concatenate, Input, Conv1D
from keras.initializers import Orthogonal
from keras.regularizers import l2, L1L2
import keras.backend as K
import numpy as np

from loaders.custom import l4_loss

def frequency_curve(fq):
    def fq_map(x):
        if x <= 300:   # minimal GSM response
            return 0.5 + x / 600.
        elif x <= 800: # hearing maximum
            return 1 + (x - 300) / 1000.
        elif x <= 1500: # hearing minimum
            return 1.5 - (x - 800) / 1400.
        elif x <= 4000: # hearing maximum
            return 1 - (x - 1500) / 10000.
        else:
            return 0.75 - (x - 4000) / 20000.
    return np.array([fq_map(x) for x in fq])

def make_frequency_loss():
    fqs = np.arange(0,257) * 8000 / 256
    weights = frequency_curve(fqs)
    mapping = K.constant(weights, shape=(1,1,256))
    def loss(y_true, y_pred):
        return mapping * (y_true - y_pred) ** 2
    return loss

def maximal_error(x, y):
    return K.max(K.abs(y - x))

class TRIM(object):
    @staticmethod
    def tight():
        # tight 4 model has the regularization and deconvolution for expanding nodes
        regularizing_value = 3e-5
        filters = 12
        Inp = Input((None, 129))
        I = Lambda(lambda x: K.expand_dims(x, axis=3))(Inp)
        L = LeakyReLU(0.5)(Conv2D(filters, kernel_size=[9,3], padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(I))
        L = LeakyReLU(0.5)(Conv2D(filters, kernel_size=[7,5], padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        M = LeakyReLU(0.5)(Conv2D(filters, kernel_size=[9,3], padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(I))
        M = LeakyReLU(0.5)(Conv2D(filters, kernel_size=[7,5], padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(M))
        I = Dropout(0.5)(concatenate([L, M]))
        L = LeakyReLU(0.5)(Conv2D(filters, kernel_size=[9,3], strides=(1,2), padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(I))
        L = LeakyReLU(0.5)(Conv2D(1, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.5)(Conv2D(1, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = UpSampling2D([1,2])(L)
        L = LeakyReLU(0.5)(Deconv2D(filters, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        M = LeakyReLU(0.5)(Deconv2D(filters, kernel_size=[7,5], padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = UpSampling2D([1,2])(Dropout(0.5)(concatenate([L, M])))
        L = LeakyReLU(0.5)(Deconv2D(filters, kernel_size=[9,3], padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.5)(Deconv2D(filters, kernel_size=[9,3], padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.5)(Deconv2D(filters, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.5)(Conv2D(1, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = Lambda(lambda x:x[:,:,1:258,:])(L)
        L = TimeDistributed(Flatten())(L)
        model = Model(Inp, L)
        model.compile('adam', 'mse', metrics=[maximal_error, 'mae'])
        model.summary()
        return model

    @staticmethod
    def tight2():
        # tight 4 model has the regularization and deconvolution for expanding nodes
        regularizing_value = 3e-5
        filters = 2048
        Inp = Input((None, 129))
        L = LeakyReLU(0.5)(Conv1D(filters, kernel_size=9, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(Inp))
        L = LeakyReLU(0.5)(Conv1D(filters, kernel_size=9, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        M = LeakyReLU(0.5)(Conv1D(filters, kernel_size=9, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(Inp))
        M = LeakyReLU(0.5)(Conv1D(filters, kernel_size=9, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(M))
        I = Dropout(0.5)(concatenate([L, M]))
        L = LeakyReLU(0.5)(Conv1D(filters, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(I))
        L = LeakyReLU(0.5)(Conv1D(filters, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.5)(Conv1D(filters, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = Lambda(lambda x:K.expand_dims(x, -1))(L)
        L = LeakyReLU(0.5)(Deconv2D(1, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        M = LeakyReLU(0.5)(Deconv2D(1, kernel_size=[9,1], padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.5)(Deconv2D(1, kernel_size=[9,1], padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.5)(Deconv2D(1, kernel_size=[9,1], padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.5)(Deconv2D(1, kernel_size=[9,1], padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = TimeDistributed(Flatten())(L)
        L = LeakyReLU(0.5)(Conv1D(257, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = Lambda(lambda x:x[:,:,:])(L)
        model = Model(Inp, L)
        model.compile('adam', 'mse', metrics=[maximal_error, 'mae'])
        model.summary()
        return model

    @staticmethod
    def smaller():
        # this model is a smaller version of tight2 while maintaining layer archotecture - we try to make it a true DAE, compressing to 64 features
        # update: it's no longer the same...
        regularizing_value = 3e-4 # since 10 times less features, then regularization may be smaller
        Inp = Input((None, 129))
        L = LeakyReLU(0.05)(Conv1D(129, kernel_size=9, kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(Inp))
        L = LeakyReLU(0.05)(Conv1D(256, kernel_size=9, kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.05)(Conv1D(256, kernel_size=9, kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.05)(Conv1D(256, kernel_size=9, kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.05)(Conv1D(192, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.05)(Conv1D(128, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.05)(Conv1D(64, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = Lambda(lambda x:K.expand_dims(x, 2))(L)
        L = LeakyReLU(0.05)(Deconv2D(128, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.05)(Deconv2D(192, kernel_size=[9,1], kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.05)(Deconv2D(256, kernel_size=[9,1], kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.05)(Deconv2D(384, kernel_size=[9,1], kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = LeakyReLU(0.05)(Deconv2D(512, kernel_size=[9,1], kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = TimeDistributed(Flatten())(L)
        L = LeakyReLU(0.05)(Conv1D(257, kernel_size=1, padding='same', kernel_initializer=Orthogonal(), activation='linear', kernel_regularizer=l2(regularizing_value))(L))
        L = Lambda(lambda x:x[:,:,:])(L)
        model = Model(Inp, L)
        model.compile('adam', 'mae', metrics=[maximal_error, 'mse', l4_loss])
        model.summary()
        return model

