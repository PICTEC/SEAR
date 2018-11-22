#!/usr/bin/env python3

import keras
import sklearn.preprocessing as skpp

import loaders
from loaders.transform import AddGaussianNoise, Windowing, GSMize, MixReverb, MixNoise, ConstantQTransform
from loaders.dataset import Dataset, enable_multiprocessing, disable_multiprocessing
from loaders.feature import LogPowerRFFTWithoutZero, Scale, Trim
from loaders.experiment import SingleExperiment
from loaders.measures import MSE, NullMSE, LogSpectralDistance, TimeDomainMSE
from loaders.schedule import Schedule
from loaders.estimator import DefaultVisualizeTransform

import loaders.extras

from loaders.CLR import CLR

import models
from models.CDAE import CDAE
from models.EHNET import EHNET
from models.SRCDAE import SRCDAE
from models.TRIM import TRIM
from models.SEP import SEP
from models.GAN import GAN_model

TRAIN = 24000
VALID = 1000
TEST = 1000

noise_class = lambda x: GSMize(x)
dataset = Dataset.from_folder("DAE-test", verbose = False, dataset_pad = 8, trim_lengths = 160000, cache = "cache", ram_cache_size = None)
input_transform = Windowing(trim_size = 4)
input_transform.add_feature(Trim(LogPowerRFFTWithoutZero(), 128))
dataset.add_input(noise_class(input_transform))
output_transform = Windowing(trim_size = 4)
output_transform.add_feature(LogPowerRFFTWithoutZero())
output_transform.add_inverse(loaders.extras.delog_griffin_lim)  # removed scaling
dataset.add_output(output_transform)
train, valid, test = dataset.iterator(8, train_data = TRAIN, valid_data = VALID, test_data= TEST)
enable_multiprocessing()


# cback = loaders.extras.LossHistory()

try:
    experiment = SingleExperiment(GAN_model, train = train, valid = valid, test = test)
    experiment.add_schedule(Schedule([0.0005], [4], lambda rate: keras.optimizers.Adam(rate, clipvalue = 1.)))
    experiment.add_measure(MSE(), name='mse')
    # experiment.add_measure(LogSpectralDistance(), name='log-spectral distance')
    # experiment.add_measure(TimeDomainMSE(input_transform), name='time-domain mse')
    experiment.add_measure(NullMSE(), name='null-experiment mse')
    experiment.run()
    experiment.report()
    experiment.build_estimator(dataset, visualizer_transform=DefaultVisualizeTransform())

finally:
    disable_multiprocessing()

"""
cite:
https://github.com/uqfoundation/pathos
"""
