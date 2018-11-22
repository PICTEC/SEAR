# Generative Adversarial Networks

from keras.layers         import Input, Conv2D, AveragePooling2D, Flatten, Dense, Deconv2D, UpSampling2D, concatenate, BatchNormalization, Lambda, LeakyReLU, TimeDistributed, Reshape, LSTM, GaussianNoise, Conv1D
from keras.models         import Model
from keras.datasets.mnist import load_data
from keras.utils          import to_categorical
from keras.initializers   import Orthogonal
from keras.regularizers   import L1L2
import keras.backend as K
import tensorflow as tf

from loaders.custom import identity_loss

import numpy as np
import random

class GAN:
    def __init__(self, generator=None, discriminator=None):
        input_lr = Input((14,14,1))
        input_hr = Input((28,28,1))
        self.generator = generator if generator else self.default_gen
        self.generator = self.generator(input_lr)
        self.discriminator = discriminator if discriminator else self.default_discr
        self.discriminator = self.discriminator(input_lr, input_hr)
        self.discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer='sgd')
        self.discriminator.trainable = False ## can I train nontrainable?
        fake_hr = self.generator.layers[-1].output
        valid_output = self.discriminator([input_lr, fake_hr])
        valid_output = valid_output[0]
        self.trainer = Model([input_lr, input_hr], [valid_output, fake_hr])
        self.trainer.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam')

    def fit_generator(self, iterator, batch_size=32, epochs=1, verbose=True):
        iterations = len(iterator) // batch_size * epochs
        for i in range(iterations):
            lres, hres, labels = next(iterator)
            fakes = self.generator.predict(lres)
            discriminator_loss_real = self.discriminator.train_on_batch(
                [lres, hres], [np.ones(batch_size), labels])
            discriminator_loss_fake = self.discriminator.train_on_batch(
                [lres, fakes], [np.zeros(batch_size), labels])
            lres, hres, labels = next(iterator)
            valid = np.ones(batch_size)
            loss = self.trainer.train_on_batch([lres, hres], [valid, hres])        
            if verbose:
                print(("{}/{}".format(i, iterations), discriminator_loss_real[1:], discriminator_loss_fake[1:], loss[1:]))

    @staticmethod
    def default_gen(inp):
        l1 = Conv2D(32, 3, activation='relu', padding='same')(inp)
        l1 = BatchNormalization()(l1)
        l1 = UpSampling2D(2)(l1)
        l2 = Conv2D(32, 3, activation='relu', padding='same')(l1)
        l2 = BatchNormalization()(l2)
        l4 = Conv2D(1, 1, activation='relu')(l2)
        return Model(inp, l4)

    @staticmethod
    def default_discr(inp_lr, inp_hr):
        l1 = Conv2D(16, 3, activation='relu', padding='same')(inp_hr)
        l2 = AveragePooling2D(2)(l1)
        l2prim = MaxPooling2D(2)(l1)
        l2 = concatenate([l2, l2prim, inp_lr])
        l3 = Conv2D(32, 3, activation='relu')(l2)
        l4 = AveragePooling2D(2)(l3)
        l4prim = MaxPooling2D(2)(l3)
        l5 = Flatten()(concatenate([l4, l4prim]))
        l6 = Dense(50, activation='relu')(l5)
        l7 = Dense(15, activation='relu')(l6)
        l8 = Dense(1, activation='sigmoid')(l7)
        l5prim = Flatten()(l4prim)
        l6prim = Dense(50, activation='relu')(l5prim)
        l7prim = Dense(25, activation='relu')(l6prim)
        l8prim = Dense(10, activation='softmax')(l7prim)
        return Model([inp_lr, inp_hr], [l8, l8prim])

    #...




"""
Pix2Pix:
    Gaussian loss produces blurry images and tackles sharp image problem

GANs learn a loss
    cGAN learns a _structured loss_ that depends on what other pixels do

Learns real and fake tuples of input and reconstruction
"""


# loss = GAN loss + l1 from inputs


# tuple: (sample_input, sample_reconstructon)
# reconstruction = self.generator.predict(sample_input)
# should be joined and shuffled
# train discriminator: self.discri.train_on_batch(sample_reconst..., np.zeros())
# train discriminator: self.discri.train_on_batch(sample_reconst..., np.ones())

# get some images, discriminate them
# train the COMBINEd model ([input, true_Recon], [valid = 1, ?])



# Idea: Patch GAN - whether pathces in image are fake or real?
# Generator with skips


# https://github.com/eriklindernoren/Keras-GAN/blob/master/srgan/srgan.py

class Generator:
    def __init__(self, *sources, batch_size=32):
        self.sources = sources
        sel = list(range(sources[0].shape[0]))
        random.shuffle(sel)
        self.selection = np.array(sel)
        self.ix = 0
        self.batch_size = batch_size

    def __next__(self):
        if self.ix + self.batch_size >= len(self.selection):
            sel = list(range(self.sources[0].shape[0]))
            random.shuffle(sel)
            self.selection = np.array(sel)
            self.ix = 0
        ret = self.selection[self.ix : self.ix + self.batch_size]
        self.ix += self.batch_size
        return [s[ret] for s in self.sources]

    def __len__(self):
        return len(self.selection)

if __name__ == 1234567:
    gan = GAN()
    
    gan.generator.summary()
    gan.discriminator.summary()
    trX, trY = load_data()[0]
    mask = np.remainder(np.arange(28), 2).astype(np.bool)
    mask = mask.reshape(-1,1) * mask.reshape(1, -1)
    mask = np.tile(mask.reshape(28, 28, 1), 60000)
    mask = np.swapaxes(mask, 0, 2)
    lr = trX[mask].reshape(60000, 14, 14).astype(np.float32).reshape(-1, 14, 14, 1)
    hr = trX.astype(np.float32).reshape(-1, 28, 28, 1)
    lr /= 256
    hr /= 256
    trY = to_categorical(trY)
    gan.fit_generator(Generator(lr, hr, trY), epochs=5)


# https://github.com/soumith/ganhacks
# - don't use ReLUs, normalize to [-1; 1]



# - Dropout on Generation?



# the discriminator has to not overfit in order for GAN to work well
# -> add some noise to training - wrong labels








def generator_of_high(input_lower):
    wrap = lambda x: BatchNormalization()(LeakyReLU(0.5)(x))
    lyr = Lambda(lambda x:K.expand_dims(x, axis=-1))(input_lower)
    # we want to extract features of different filters of spectrogram
    # we would also like to generalize on time to reduce dimensionality
    time1 = wrap(Conv2D(8, kernel_size=(3, 1), strides=(2, 1), padding='same', activation=None)(lyr))
    time2 = wrap(Conv2D(8, kernel_size=(3, 1), strides=(2, 1), padding='same', activation=None)(time1))
    # now from changes in time we can deduce filters in frequency
    freq1 = wrap(Conv2D(16, kernel_size=(1, 3), strides=(1,2), activation=None)(time2))
    freq2 = wrap(Conv2D(16, kernel_size=(1, 3), strides=(1,2), activation=None)(freq1))
    # now we have compressed the frequency characteristics, need to encode what kind
    # of noise generate in higher frequencies
    hidden1 = TimeDistributed(Flatten())(freq2)
    hidden2 = wrap(TimeDistributed(Dense(128))(hidden1))
    hidden3 = wrap(TimeDistributed(Dense(64))(hidden2))
    hidden4 = Lambda(lambda x:K.expand_dims(x, axis=-1))(hidden3)
    # generation of higher frequencies
    # expand in time twice (x4)
    # expand in frequency once (x2)      
    freqT1 = wrap(Deconv2D(16, kernel_size=(1, 3), strides=(1,2), activation=None)(hidden4))
    freqT1 = Lambda(lambda x: x[:, :, 1:, :])(freqT1)
    timeT1 = wrap(Deconv2D(8, kernel_size=(3, 1), strides=(2, 1), padding='same', activation=None)(freqT1))
    timeT2 = wrap(Deconv2D(8, kernel_size=(3, 1), strides=(2, 1), padding='same', activation=None)(timeT1))
    final = Deconv2D(1, 1, activation=None)(timeT2)
    final = Lambda(lambda x: K.sum(x, axis=-1))(final) # error here
    return Model(input_lower, concatenate([input_lower, final]))

def GAN_model():
    mdl = generator_of_high(Input((None, 128)))
    mdl.compile('adam', 'mae')
    return mdl

def GAN_model_mse():
    mdl = generator_of_high(Input((None, 128)))
    mdl.compile('adam', 'mse')
    return mdl

def with_speech_model():
    input_lower = Input((None, 128))
    wrap = lambda x: LeakyReLU(0.5)(x)
    lyr = Lambda(lambda x:K.expand_dims(x, axis=-1))(input_lower)
    # we want to extract features of different filters of spectrogram
    # we would also like to generalize on time to reduce dimensionality
    time1 = wrap(Conv2D(8, kernel_size=(3, 1), strides=(2, 1), padding='same', activation=None)(lyr))
    time2 = wrap(Conv2D(8, kernel_size=(3, 1), strides=(2, 1), padding='same', activation=None)(time1))
    # now from changes in time we can deduce filters in frequency
    freq1 = wrap(Conv2D(16, kernel_size=(1, 3), strides=(1,2), activation=None)(time2))
    freq2 = wrap(Conv2D(16, kernel_size=(1, 3), strides=(1,2), activation=None)(freq1))
    # now we have compressed the frequency characteristics, need to encode what kind
    # of noise generate in higher frequencies
    hidden1 = TimeDistributed(Flatten())(freq2)
    hidden2 = wrap(TimeDistributed(Dense(128))(hidden1))
    hidden3 = LSTM(64, return_sequences=True, kernel_initializer=Orthogonal())(hidden2)
    hidden4mean = LSTM(64, return_sequences=True, kernel_initializer=Orthogonal())(hidden3)
    hidden4dev = LSTM(64, return_sequences=True, kernel_initializer=Orthogonal())(hidden3)
    hidden5 = Lambda(lambda x: x[0] + x[1] * tf.random_normal(tf.shape(x[1])))([hidden4mean, hidden4dev])
    hidden6 = Lambda(lambda x:K.expand_dims(x, axis=-1))(hidden5)
    # generation of higher frequencies
    # expand in time twice (x4)
    # expand in frequency once (x2)      
    freqT1 = wrap(Deconv2D(16, kernel_size=(1, 3), strides=(1,2), activation=None)(hidden6))
    freqT1 = Lambda(lambda x: x[:, :, 1:, :])(freqT1)
    timeT1 = wrap(Deconv2D(8, kernel_size=(3, 1), strides=(2, 1), padding='same', activation=None)(freqT1))
    timeT2 = wrap(Deconv2D(8, kernel_size=(3, 1), strides=(2, 1), padding='same', activation=None)(timeT1))
    final = Deconv2D(1, 1, activation=None)(timeT2)
    final = Lambda(lambda x: K.sum(x, axis=-1))(final) # error here
    mdl = Model(input_lower, concatenate([input_lower, final]))
    mdl.compile('adam', 'mae')
    mdl.summary()
    return mdl

def speech_model_discriminator(input_lower, input_upper):
    wrap = lambda x: LeakyReLU(0.5)(x)
    input_upper2 = Lambda(lambda x: x[:,:,128:])(input_upper)
    lyr = concatenate([input_lower, input_upper2])
    lyr = Lambda(lambda x:K.expand_dims(x, axis=-1))(lyr)
    # we want to extract features of different filters of spectrogram
    # we would also like to generalize on time to reduce dimensionality
    time1 = wrap(Conv2D(8, kernel_size=(3, 1), strides=(2, 1), padding='same', activation=None)(lyr))
    time2 = wrap(Conv2D(8, kernel_size=(3, 1), strides=(2, 1), padding='same', activation=None)(time1))
    # now from changes in time we can deduce filters in frequency
    freq1 = wrap(Conv2D(16, kernel_size=(1, 3), strides=(1,2), activation=None)(time2))
    freq2 = wrap(Conv2D(16, kernel_size=(1, 3), strides=(1,2), activation=None)(freq1))
    # now we have compressed the frequency characteristics, need to encode what kind
    # of noise generate in higher frequencies
    hidden1 = TimeDistributed(Flatten())(freq2)
    hidden2 = wrap(TimeDistributed(Dense(128))(hidden1))
    hidden3 = LSTM(64, return_sequences=True, kernel_initializer=Orthogonal())(hidden2)
    hidden4 = LSTM(16, return_sequences=True, kernel_initializer=Orthogonal())(hidden3)
    authenticity = LSTM(1, activation='sigmoid', kernel_initializer=Orthogonal())(hidden4)
    return Model([input_lower, input_upper], authenticity)






def librispeech_recog_model(n_phonemes, label_lengths):
    def mk_model():
        input_lower = Input((None, 128))
        wrap = lambda x: LeakyReLU(0.5)(x)
        lyr = Lambda(lambda x: K.expand_dims(x, axis=-1))(input_lower)
        time1 = wrap(Conv2D(32, kernel_size=(5, 1), strides=(4, 1), padding='same', activation=None)(lyr))
        freq1 = wrap(Conv2D(40, kernel_size=(1, 5), strides=(1, 4), activation=None)(time1))
        hidden1 = TimeDistributed(Flatten())(freq1)
        hidden2 = wrap(TimeDistributed(Dense(512))(hidden1))
        hidden3 = LSTM(256, return_sequences=True, kernel_initializer=Orthogonal())(hidden2)

        # CTC loss function - calculated in the graph
        phonemes = LSTM(n_phonemes + 1, return_sequences=True, kernel_initializer=Orthogonal(), activation='sigmoid')(hidden3)
        def ctc_loss_function(arguments):
            y_pred, y_true, input_length, label_length = arguments
            return K.ctc_batch_cost(y_true, y_pred, input_length // 4, label_length)
        label_input = Input(shape = (label_lengths + 1,))
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_lambda = Lambda(ctc_loss_function, output_shape=(1,), name='ctc')([phonemes, label_input, input_length, label_length])
    
        hidden4mean = LSTM(128, return_sequences=True, kernel_initializer=Orthogonal())(hidden3)
        hidden4dev = LSTM(128, return_sequences=True, kernel_initializer=Orthogonal())(hidden3)
        hidden5 = Lambda(lambda x: x[0] + x[1] * tf.random_normal(tf.shape(x[1])))([hidden4mean, hidden4dev])
        hidden6 = Lambda(lambda x: K.expand_dims(x, axis=-1))(hidden5)
        freqT1 = wrap(Deconv2D(12, kernel_size=(1, 5), strides=(1,1), activation=None)(hidden6))
        freqT2 = Lambda(lambda x: x[:, :, 1:129, :])(freqT1)
        timeT1 = wrap(Deconv2D(12, kernel_size=(5, 1), strides=(4, 1), padding='same', activation=None)(freqT2))
        final = Conv2D(1, 1, activation=None)(timeT1)
        final = Lambda(lambda x: K.sum(x, axis=-1))(final) # error here
    
        mdl = Model([input_lower, label_input, input_length, label_length], 
            [concatenate([input_lower, final]), loss_lambda])
        mdl.compile('adam', ['mae', identity_loss])
        mdl.summary()
        return mdl
    return mk_model


def simplest_model():
    input_lower = Input((None, 128))
    wrap = lambda x: LeakyReLU(0.5)(x)
    final = wrap(TimeDistributed(Dense(256))(input_lower))
    final = wrap(TimeDistributed(Dense(64))(final))
    final = wrap(TimeDistributed(Dense(256))(final))
    final = wrap(TimeDistributed(Dense(128))(final))
    model = Model(input_lower, concatenate([input_lower, final]))
    model.summary()
    model.compile('adam', 'mae')
    return model

def old_dual_model():
    input_lower = Input((None, 129))
    wrap = lambda x: LeakyReLU(0.5)(x)
    final = wrap(Conv1D(2048, 17)(input_lower))
    final = wrap(TimeDistributed(Dense(512))(input_lower))
    final = wrap(TimeDistributed(Dense(512))(final))
    final = wrap(TimeDistributed(Dense(64))(final))
    final = wrap(TimeDistributed(Dense(128))(final))
    reconstructed_lower = wrap(Conv1D(2048, 9)(input_lower))
    reconstructed_lower = wrap(Conv1D(1024, 9)(input_lower))
    reconstructed_lower = wrap(Dense(64)(input_lower))
    reconstructed_lower = wrap(Dense(129)(input_lower))
    model = Model(input_lower, concatenate([reconstructed_lower, final]))
    model.summary()
    model.compile('adam', 'mae')
    return model

def dual_model():
    # make it deeper!
    input_lower = Input((None, 129))
    wrap = lambda x: LeakyReLU(0.05)(x)
    final = wrap(Conv1D(4096, 17, padding='same', kernel_regularizer=L1L2(l2=5e-7, l1=1e-8))(input_lower))
    final = wrap(TimeDistributed(Dense(2048, kernel_regularizer=L1L2(l2=5e-7, l1=1e-8)))(final))
    final = wrap(TimeDistributed(Dense(2048, kernel_regularizer=L1L2(l2=5e-7, l1=1e-8)))(final))
    final = wrap(TimeDistributed(Dense(1024, kernel_regularizer=L1L2(l2=5e-7, l1=1e-8)))(final))
    final = wrap(TimeDistributed(Dense(1024, kernel_regularizer=L1L2(l2=5e-7, l1=1e-8)))(final))
    final = wrap(TimeDistributed(Dense(512, kernel_regularizer=L1L2(l2=1e-6, l1=1e-8)))(final))
    final = wrap(TimeDistributed(Dense(512, kernel_regularizer=L1L2(l2=1e-6, l1=1e-8)))(final))
    final = wrap(TimeDistributed(Dense(256, kernel_regularizer=L1L2(l2=1e-6)))(final))
    final = wrap(TimeDistributed(Dense(256, kernel_regularizer=L1L2(l2=1e-6)))(final))
    final = wrap(TimeDistributed(Dense(128))(final))
    reconstructed_lower = wrap(Conv1D(4096, 9, padding='same', kernel_regularizer=L1L2(l2=5e-7, l1=1e-8))(input_lower))
    reconstructed_lower = wrap(Conv1D(2048, 9, padding='same', kernel_regularizer=L1L2(l2=5e-7, l1=1e-8))(reconstructed_lower))
    reconstructed_lower = wrap(Dense(1024, kernel_regularizer=L1L2(l2=5e-7, l1=1e-8))(reconstructed_lower))
    reconstructed_lower = wrap(Dense(1024, kernel_regularizer=L1L2(l2=5e-7, l1=1e-8))(reconstructed_lower))
    reconstructed_lower = wrap(Dense(512, kernel_regularizer=L1L2(l2=1e-6, l1=1e-8))(reconstructed_lower))
    reconstructed_lower = wrap(Dense(512, kernel_regularizer=L1L2(l2=1e-6, l1=1e-8))(reconstructed_lower))
    reconstructed_lower = wrap(Dense(256, kernel_regularizer=L1L2(l2=1e-6))(reconstructed_lower))
    reconstructed_lower = wrap(Dense(256, kernel_regularizer=L1L2(l2=1e-6))(reconstructed_lower))
    reconstructed_lower = wrap(Dense(129)(reconstructed_lower))
    model = Model(input_lower, concatenate([reconstructed_lower, final]))
    model.summary()
    model.compile('adam', 'mae')
    return model

def discriminator(input, output):
    wrap = lambda x: LeakyReLU(0.1)(x)
    lfb = Lambda(lambda x: x[:,:,:129])(output)
    hfb = Lambda(lambda x: x[:,:,129:])(output)
    exp1 = Lambda(lambda x:K.expand_dims(x, axis=-1))(input)
    exp2 = Lambda(lambda x:K.expand_dims(x, axis=-1))(lfb)
    exp_hfb = Lambda(lambda x:K.expand_dims(x, axis=-1))(hfb)
    c = concatenate([exp1, exp2])
    time1 = wrap(Conv2D(32, kernel_size=(17, 1), strides=(4, 1), padding='same', activation=None)(c))
    freq1 = wrap(Conv2D(32, kernel_size=(1, 5), activation=None)(time1))
    freq1 = TimeDistributed(Flatten())(freq1)
    hidden_lfb = wrap(TimeDistributed(Dense(2048))(freq1))
    #hidden_lfb = wrap(TimeDistributed(Dense(1024))(hidden_lfb))
    #hidden_lfb = wrap(TimeDistributed(Dense(1024))(hidden_lfb))
    #hidden_lfb = wrap(TimeDistributed(Dense(512))(hidden_lfb))
    hidden_lfb = TimeDistributed(Dense(512, activation='tanh'))(hidden_lfb)
    hidden_lfb = TimeDistributed(Dense(256, activation='tanh'))(hidden_lfb)
    time1 = wrap(Conv2D(32, kernel_size=(17, 1), strides=(4, 1), padding='same', activation=None)(exp1))
    freq1 = wrap(Conv2D(48, kernel_size=(1, 5), strides=(1, 2), activation=None)(time1))
    freq1 = TimeDistributed(Flatten())(freq1)
    time2 = wrap(Conv2D(32, kernel_size=(17, 1), strides=(4, 1), padding='same', activation=None)(exp_hfb))
    freq2 = wrap(Conv2D(48, kernel_size=(1, 5), strides=(1, 2), activation=None)(time2))
    freq2 = TimeDistributed(Flatten())(freq2)
    hidden_hfb = concatenate([freq1, freq2])
    hidden_hfb = wrap(TimeDistributed(Dense(2048))(hidden_hfb))
    #hidden_hfb = wrap(TimeDistributed(Dense(1024))(hidden_hfb))
    #hidden_hfb = wrap(TimeDistributed(Dense(1024))(hidden_hfb))
    #hidden_hfb = wrap(TimeDistributed(Dense(512))(hidden_hfb))
    hidden_hfb = TimeDistributed(Dense(512, activation='tanh'))(hidden_hfb)
    hidden_hfb = TimeDistributed(Dense(256, activation='tanh'))(hidden_hfb)
    hidden = concatenate([hidden_lfb, hidden_hfb])
    hidden = TimeDistributed(Dense(1))(hidden)
    authenticity = Lambda(lambda x:K.sigmoid(K.mean(x, axis=1)))(hidden)
    model = Model([input, output], authenticity)
    model.compile('adam', 'binary_crossentropy')
    return model

def baseline():
    input_lower = Input((None, 129))
    convo = Conv1D(4096, 17, activation='tanh', padding='same')(input_lower)
    final = TimeDistributed(Dense(2048, activation='tanh'))(convo)
    final = TimeDistributed(Dense(128))(final)
    reconstructed_lower = Dense(2048, activation='tanh')(convo)
    reconstructed_lower = Dense(129)(reconstructed_lower)
    model = Model(input_lower, concatenate([reconstructed_lower, final]))
    model.summary()
    model.compile('adam', 'mse')
    return model

def crippled_discriminator(input, output):
    wrap = lambda x: LeakyReLU(0.1)(x)
    lfb = Lambda(lambda x: x[:,:,:129])(output)
    hfb = Lambda(lambda x: x[:,:,129:])(output)
    exp1 = Lambda(lambda x:K.expand_dims(x, axis=-1))(input)
    exp2 = Lambda(lambda x:K.expand_dims(x, axis=-1))(lfb)
    exp_hfb = Lambda(lambda x:K.expand_dims(x, axis=-1))(hfb)
    c = concatenate([exp1, exp2])
    time1 = wrap(Conv2D(8, kernel_size=(17, 1), strides=(4, 1), padding='same', activation=None)(c))
    freq1 = wrap(Conv2D(8, kernel_size=(1, 5), activation=None)(time1))
    freq1 = TimeDistributed(Flatten())(freq1)
    hidden_lfb = wrap(TimeDistributed(Dense(256))(freq1))
    hidden_lfb = TimeDistributed(Dense(64, activation='tanh'))(hidden_lfb)
    time1 = wrap(Conv2D(8, kernel_size=(17, 1), strides=(4, 1), padding='same', activation=None)(exp1))
    freq1 = wrap(Conv2D(8, kernel_size=(1, 5), strides=(1, 2), activation=None)(time1))
    freq1 = TimeDistributed(Flatten())(freq1)
    time2 = wrap(Conv2D(8, kernel_size=(17, 1), strides=(4, 1), padding='same', activation=None)(exp_hfb))
    freq2 = wrap(Conv2D(8, kernel_size=(1, 5), strides=(1, 2), activation=None)(time2))
    freq2 = TimeDistributed(Flatten())(freq2)
    hidden_hfb = concatenate([freq1, freq2])
    hidden_hfb = wrap(TimeDistributed(Dense(256))(hidden_hfb))
    hidden_hfb = TimeDistributed(Dense(64, activation='tanh'))(hidden_hfb)
    hidden = concatenate([hidden_lfb, hidden_hfb])
    hidden = TimeDistributed(Dense(1))(hidden)
    authenticity = Lambda(lambda x:K.sigmoid(K.mean(x, axis=1)))(hidden)
    model = Model([input, output], authenticity)
    model.compile('adam', 'binary_crossentropy')
    return model
