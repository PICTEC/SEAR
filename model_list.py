import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Lambda, LeakyReLU, TimeDistributed, Flatten, BatchNormalization, concatenate
import keras.backend as K
import keras.callbacks as kc

def mk_dense(variant):
    assert variant in ['trim', 'full_separate', 'full_whole']
    def mk_model(**kwargs):
        wrap = lambda x: LeakyReLU(0.01)(x)
        input_lower = Input((None, 129), name="input_lf")
        layer = Lambda(lambda x: x / kwargs["MAX"])(input_lower)
        lyr = Lambda(K.expand_dims)(layer)
        windowing = Conv2D(15, (15, 1), padding='same', use_bias=False)
        lyr = windowing(lyr)
        windowing.set_weights([np.eye(15).reshape(15, 1, 1, 15)])
        windowing.trainable = False
        base = lyr = TimeDistributed(Flatten())(lyr)
        if variant == "trim":
            lyr = wrap(Dense(2048)(lyr))
            lyr = wrap(Dense(2048)(lyr))
            lf_and_hf = Dense(129)(lyr)
        if variant == "full_separate":
            lyr = wrap(Dense(1024)(base))
            lyr = wrap(Dense(1024)(lyr))
            out1 = Dense(129)(lyr)
            lyr = wrap(Dense(1024)(base))
            lyr = wrap(Dense(1024)(lyr))
            out2 = Dense(128)(lyr)
            lf_and_hf = concatenate([out1, out2])
        if variant == "full_whole":
            lyr = wrap(Dense(2048)(lyr))
            lyr = wrap(Dense(2048)(lyr))
            lf_and_hf = Dense(257)(lyr)
        lf_and_hf = Lambda(lambda x: x * kwargs["MAX"])(lf_and_hf)
        mdl = Model(input_lower, lf_and_hf)
        mdl.summary()
        mdl.compile('adam', 'mse')
        return mdl
    return mk_model

def mk_conv_dae(variant, complexity):
    assert variant in ['trim', 'full_separate', 'full_whole']
    def mk_model(**kwargs):
        input_lower = Input((None, 129), name="input_lf")
        layer = Lambda(lambda x: x / kwargs["MAX"])(input_lower)
        base = Lambda(K.expand_dims)(layer)
        if variant == "trim":
            layer = base
            for i in range(complexity):
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 20, kernel_size=(1,5), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 24, kernel_size=(9,1), padding='same', activation='linear')(layer))
            layer = LeakyReLU(0.01)(Conv2D(1, kernel_size=(1,1), padding='same', activation='linear')(layer))
            for i in range(complexity - 1, -1, -1):
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 24, kernel_size=(9,1), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 20, kernel_size=(1,5), padding='same', activation='linear')(layer))     
            lf_and_hf = Dense(129)(TimeDistributed(Flatten())(layer))
        if variant == "full_separate":
            outs = []
            for sz in [129, 128]:
                layer = base
                for i in range(complexity):
                    layer = LeakyReLU(0.01)(Conv2D(i * 4 + 10, kernel_size=(1,5), padding='same', activation='linear')(layer))
                    layer = LeakyReLU(0.01)(Conv2D(i * 4 + 12, kernel_size=(9,1), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(1, kernel_size=(1,1), padding='same', activation='linear')(layer))
                for i in range(complexity - 1, -1, -1):
                    layer = LeakyReLU(0.01)(Conv2D(i * 4 + 12, kernel_size=(9,1), padding='same', activation='linear')(layer))
                    layer = LeakyReLU(0.01)(Conv2D(i * 4 + 10, kernel_size=(1,5), padding='same', activation='linear')(layer))
                outs.append(Dense(sz)(TimeDistributed(Flatten())(layer)))
            lf_and_hf = concatenate(outs)
        if variant == "full_whole":
            layer = base
            for i in range(complexity):
                layer = LeakyReLU(0.01)(Conv2D(i * 4 + 10, kernel_size=(1,5), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(i * 4 + 12, kernel_size=(9,1), padding='same', activation='linear')(layer))
            layer = LeakyReLU(0.01)(Conv2D(1, kernel_size=(1,1), padding='same', activation='linear')(layer))
            for i in range(complexity - 1, -1, -1):
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 24, kernel_size=(9,1), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 20, kernel_size=(1,5), padding='same', activation='linear')(layer))
            lf_and_hf = Dense(257)(TimeDistributed(Flatten())(layer))            
        lf_and_hf = Lambda(lambda x: x * kwargs["MAX"])(lf_and_hf)
        mdl = Model(input_lower, lf_and_hf)
        mdl.summary()
        mdl.compile('adam', 'mse')
        return mdl
    return mk_model

def mk_conv_dense(variant, complexity):
    assert variant in ['trim', 'full_separate', 'full_whole']
    def mk_model(**kwargs):
        input_lower = Input((None, 129), name="input_lf")
        layer = Lambda(lambda x: x / kwargs["MAX"])(input_lower)
        base = Lambda(K.expand_dims)(layer)
        if variant == "trim":
            layer = base
            for i in range(complexity):
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 20, kernel_size=(1,5), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(i * 8 + 24, kernel_size=(9,1), padding='same', activation='linear')(layer))
            layer = TimeDistributed(Flatten())(layer)
            layer = LeakyReLU(0.01)(Dense(1024)(layer))
            hidden = layer = LeakyReLU(0.01, name='hidden')(Dense(96)(layer))
            for i in range(complexity - 1, -1, -1):
                layer = LeakyReLU(0.01)(Dense(512)(layer))
            lf_and_hf = Dense(129)(layer)
        if variant == "full_separate":
            outs = []
            for sz in [129, 128]:
                layer = base
                for i in range(complexity):
                    layer = LeakyReLU(0.01)(Conv2D(i * 4 + 10, kernel_size=(1,5), padding='same', activation='linear')(layer))
                    layer = LeakyReLU(0.01)(Conv2D(i * 4 + 12, kernel_size=(9,1), padding='same', activation='linear')(layer))
                layer = TimeDistributed(Flatten())(layer)
                layer = LeakyReLU(0.01)(Dense(512)(layer))
                layer = LeakyReLU(0.01)(Dense(96)(layer))
                for i in range(complexity - 1, -1, -1):
                    layer = LeakyReLU(0.01)(Dense(512)(layer))
                outs.append(Dense(sz)(layer))
            lf_and_hf = concatenate(outs)
        if variant == "full_whole":
            layer = base
            for i in range(complexity):
                layer = LeakyReLU(0.01)(Conv2D(i * 4 + 10, kernel_size=(1,5), padding='same', activation='linear')(layer))
                layer = LeakyReLU(0.01)(Conv2D(i * 4 + 12, kernel_size=(9,1), padding='same', activation='linear')(layer))
            layer = TimeDistributed(Flatten())(layer)
            layer = LeakyReLU(0.01)(Dense(1024)(layer))
            hidden = layer = LeakyReLU(0.01, name='hidden')(Dense(96)(layer))
            for i in range(complexity - 1, -1, -1):
                layer = LeakyReLU(0.01)(Dense(512)(layer))
            lf_and_hf = Dense(257)(layer)
        lf_and_hf = Lambda(lambda x: x * kwargs["MAX"])(lf_and_hf)
        mdl = Model(input_lower, lf_and_hf)
        mdl.summary()
        mdl.compile('adam', 'mse')
        return mdl
    return mk_model

def get_schema(model_type, variant):
    if model_type == "dense":
        return mk_dense(variant)
    if model_type.startswith("conv_dense"):
        return mk_conv_dense(variant, 2 if model_type == "conv_dense" else 4)
    if model_type.startswith("dae"):
        return mk_conv_dae(variant, 2 if model_type == "dae" else 4)