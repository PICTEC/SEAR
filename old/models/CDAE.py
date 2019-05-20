from keras.models import Sequential, Model
from keras.layers import Conv2D, Lambda, MaxPooling2D, UpSampling2D, TimeDistributed, Flatten, LeakyReLU, Dropout, concatenate, Input
import keras.backend as K

class CDAE(object):
    @staticmethod
    def single_layer():
        """
        A la (Grais 2017) https://arxiv.org/pdf/1703.08019.pdf
        """
        model = Sequential()
        model.add(Lambda(lambda x: K.expand_dims(x, axis=3), input_shape = (None, 256)))
        model.add(Conv2D(12, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D((3,2)))
        model.add(Conv2D(24, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D((3,2)))
        model.add(Conv2D(36, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(42, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(36, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(Conv2D(24, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(UpSampling2D((3,2)))
        model.add(Conv2D(12, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(UpSampling2D((3,2)))
        model.add(Conv2D(1, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(TimeDistributed(Flatten()))
        model.compile('adam', 'mse')
        model.summary()
        return model

    @staticmethod
    def better_stacked_with_highway(times=3):
        Inp = Input((None, 256))
        I = Lambda(lambda x: K.expand_dims(x, axis=3))(Inp)
        filters = 36
        kernel_size = 3
        pooling_size = (9,2)
        for i in range(times):
            L = Conv2D(filters, kernel_size=kernel_size, padding='same')(I)
            L = LeakyReLU(0.5)(L)
            L = Dropout(0.25)(L)
            L = MaxPooling2D(pooling_size)(L)
            L = Conv2D(filters, kernel_size=kernel_size, padding='same')(L)
            L = LeakyReLU(0.5)(L)
            L = Dropout(0.25)(L)
            L = UpSampling2D(pooling_size)(L)
            L = Conv2D(1, kernel_size=1, padding='same')(L)
            L = LeakyReLU(0.5)(L)
            I = concatenate([L, I])
        L = Conv2D(1, kernel_size=1, padding='same')(I)
        L = LeakyReLU(0.5)(L)
        L = TimeDistributed(Flatten())(L)
        model = Model(Inp, L)
        model.compile('adam', 'mse')
        model.summary()
        return model

