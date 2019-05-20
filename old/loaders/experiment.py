import keras.backend as K
from keras.models         import Model
from keras.layers         import Input

import atexit
import logging
import numpy as np
import os
import pickle
import scipy.io.wavfile as sio
import time

import loaders.estimator
from loaders.utils import ProgressBar

class Experiment(object):
    def run(self):
        raise NotImplementedError

    def add_measure(self, measure, name=None):
        if name is None:
            name = "measure_" + str(len(self.measures))
        self.measures[name] = measure

    def add_schedule(self, schedule):
        self.schedule = schedule


class SingleExperiment(Experiment):
    """
    A wrapper around model that's designed to log results of the training and is responsible
    for logging of all the experiments.
    """
    def __init__(self, model_constructor, train, valid, test, run_parameters = None, run_path="./runs", name=None):
        self.model_constructor = model_constructor
        self.model = None
        self.test_model = None
        self.train_data = train
        self.valid_data = valid if test is not None else None # (nullable)
        self.test_data = test if test is not None else valid
        self.run_parameters = run_parameters if run_parameters is not None else {"epochs": 100}
        self.run_path = run_path
        self.measures = {}
        self.schedule = None
        self.name = name
        self.callbacks = []

    def run(self):
        if self.model is not None:
            raise RuntimeError("The experiment has been already run. To repeat it, please use remake() method to produce a new experiment")
        self.start_time = time.time()
        K.set_learning_phase(1)
        self.model = self.model_constructor()
        self.history = None
        self.results = {}
        self.measures = {k:v(self.model) for k,v in self.measures.items()} # map self measures to direct instances of measures
        trainable = self.schedule.add_model(self.model) if self.schedule is not None else self.model
        if self.name is None:
            self.name = os.path.join(self.run_path, str(self.start_time))
        try:
            os.mkdir(self.name)
        except FileExistsError:
            raise RuntimeError("Cannot save the model under the existing name")
        if self.callbacks:
            self.run_parameters["callbacks"] = self.callbacks
        try:
            if not isinstance(self.train_data, tuple):
                data_len = self.train_data.__len__()
                if self.valid_data is not None:
                    val_steps = self.valid_data.__len__()
                    hist = trainable.fit_generator(self.train_data, data_len, validation_data=self.valid_data, validation_steps = val_steps, **self.run_parameters)
                else:
                    hist = trainable.fit_generator(self.train_data, data_len, **self.run_parameters)
            else:
                trainX, trainY = self.train_data
                if self.valid_data is not None:
                    hist = trainable.fit(trainX, trainY, validation_data=self.valid_data, **self.run_parameters)
                else:
                    hist = trainable.fit(trainX, trainY, **self.run_parameters)
            logging.info("Testing...")
            K.set_learning_phase(0)
            self.history = hist.history
            if type(self.test_data) != tuple:
                self.test_data = [next(self.test_data) for i in range(len(self.test_data))]
                for name, measure in self.measures.items():
                    mean = []
                    for data_item in self.test_data:
                        mean.append(measure(*data_item)) # what kinds of are required?
                    self.results[name] = np.array(mean).mean()
            else:
                for name, measure in self.measures.items():
                    self.results[name] = measure(*self.test_data) # what kinds of are required?
        finally:
            if self.test_model is not None:
                self.test_model = self.test_model()
            self._save()

    def crossvalidate(self, times = 10):
        """
        Run a certain number of the experiments and gets the results
        """
        if self.model is not None:
            raise RuntimeError("The experiment has been already run. To repeat it, please use remake() method to produce a new experiment")
        raise NotImplementedError # TODO

    def _save(self):
        try:
            os.mkdir(self.run_path)
        except OSError:
            pass
        self.model.save(os.path.join(self.name, "model.h5"))
        if self.test_model is not None:
            self.test_model.save(os.path.join(self.name, "test_model.h5"))
        with open(os.path.join(self.name, "results.pkl"), "wb") as f:
            pickle.dump(self.results, f)
        with open(os.path.join(self.name, "history.pkl"), "wb") as f:
            pickle.dump(self.history, f)
        atexit.register(lambda: print(self.name))
        # TODO: add dumping experiment description allowing to reproduce the results

    def remake(self):
        raise NotImplementedError # TODO

    def dump(self, detransformer, train = False, valid = False, test = True):
        # TODO: changeable sample_rate
        assert self.name is not None
        path = os.path.join(self.name, "wav_data")
        if not os.path.exists(path):
            os.mkdir(path)
        datasets = []
        if train: datasets.append(self.train_data)
        if valid: datasets.append(self.valid_data)
        if test:  datasets.append(self.test_data)
        for ix, dataset in enumerate(datasets):
            n_examples = dataset[0].shape[0]
            hypo = self.model.predict(dataset[0])
            for example in range(n_examples):
                name = os.path.join(path, "{}-{}".format(ix, example))
                train_example = (detransformer(dataset[0][example, :, :]) * (2**15)).astype(np.int16)
                gt_example = (detransformer(dataset[1][example, :, :]) * (2**15)).astype(np.int16)
                hypo_example = (detransformer(hypo[example, :, :]) * (2**15)).astype(np.int16)
                sio.write("{}-train.wav".format(name), 16000, train_example)
                sio.write("{}-gt.wav".format(name), 16000, gt_example)
                sio.write("{}-hypo.wav".format(name), 16000, hypo_example)

    def report(self):
        print(self.results)

    def build_estimator(self, dataset, visualizer_transform = None):
        """
        Prototype must be gotten from dataset separately
        """
        item = dataset.outputs[0][1]
        reverse_transform = lambda x:item.inverse(x)
        dataset_prototype = dataset.make_prototype()
        estimator = loaders.estimator.Estimator(
            visualizer_transform,
            reverse_transform,
            dataset_prototype,
            self.model
        )
        estimator.save(self.name)

    def set_test_time(self, inputs, outputs):
        def mk_test_model():
            logging.info("Building test time model")
            layers = {i.name: i for i in self.model.layers}
            return Model(
                layers[inputs].output if isinstance(inputs, str) else [layers[x].output for x in inputs],
                layers[outputs].output if isinstance(outputs, str) else [layers[x].output for x in outputs])
        self.test_model = mk_test_model

    def add_callback(self, cb):
        self.callbacks.append(cb)


class Crossvalidation(Experiment):
    """
    Not to be constructed directly
    """
    pass


class GANExperiment(SingleExperiment):
    def __init__(self, generator_constructor, discriminator_constructor,
                 train, valid, test, run_parameters=None, run_path="./runs", name=None, adaptive=False,
                 discriminator_optimizer='sgd', trainer_optimizer='adam', reconstruction_loss='mae'):
        super().__init__(generator_constructor, train, valid, test, run_parameters = None, run_path="./runs", name=name)
        self.discriminator_constructor = discriminator_constructor
        self.run_parameters = run_parameters
        self.adaptive = adaptive
        self.D_limit = 1.
        self.D_limit_update = 1
        self.G_limit = 0.5
        self.G_limit_update = 1
        self.discriminator_optimizer = discriminator_optimizer
        self.trainer_optimizer = trainer_optimizer
        self.reconstruction_loss = reconstruction_loss

    def run(self):
        if self.model is not None:
            raise RuntimeError("The experiment has been already run. To repeat it, please use remake() method to produce a new experiment")
        self.start_time = time.time()
        K.set_learning_phase(1)
        self.model = self.model_constructor()
        input_lr = self.model.input
        shape = self.model.layers[-1].output_shape
        input_hr = Input(shape[1:])
        self.discriminator = self.discriminator_constructor(input_lr, input_hr)
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=self.discriminator_optimizer)
        self.discriminator.trainable = False
        fake_hr = self.model.layers[-1].output
        valid_output = self.discriminator([input_lr, fake_hr])
        self.trainer = Model([input_lr, input_hr], [valid_output, fake_hr])
        self.trainer.compile(loss=['binary_crossentropy', self.reconstruction_loss], optimizer=self.trainer_optimizer)
        self.history = None
        self.results = {}
        self.measures = {k:v(self.model) for k,v in self.measures.items()} # map self measures to direct instances of measures
        if self.name is None:
            self.name = os.path.join(self.run_path, str(self.start_time))
        try:
            os.mkdir(self.name)
        except FileExistsError:
            raise RuntimeError("Cannot save the model under the existing name")
        if self.schedule is not None:
            raise NotImplementedError("Schedule training mode is unavailable")
        try:
            try:
                batch_size = self.run_parameters["batch_size"]
            except KeyError:
                batch_size = 10
            try:
                verbose = self.run_parameters["verbose"]
            except KeyError:
                verbose = True
            try:
                epochs = self.run_parameters["epochs"]
            except KeyError:
                epochs = 1
            self.history = {}
            self.history["discriminator_loss_real"] = []
            self.history["discriminator_loss_fake"] = []
            self.history["loss"] = []
            iterations = len(self.train_data) * epochs
            self.ONES = np.ones([batch_size, 1])
            self.ONESFLAT = np.ones(batch_size)
            self.batch_size = batch_size
            self.discriminator.summary()
            self.discriminator._make_train_function()
            self.trainer._make_train_function()
            if verbose:
                pb = ProgressBar()
                pb.set_max(iterations)
            if self.adaptive:
                discriminator_loss_real, discriminator_loss_fake = [5], [5]
                loss = [0, 30, 0]
                iterations *= 2
                turns = 0
                pb.set_max(iterations)
                for i in range(iterations):
                    if sum([sum(discriminator_loss_real), sum(discriminator_loss_fake)]) > self.D_limit:
                        discriminator_loss_real, discriminator_loss_fake = self._discriminator_step()
                        step = "D"
                    elif loss[1] > self.G_limit:
                        loss = self._generator_step()
                        step = "G"
                    else:
                        discriminator_loss_real, discriminator_loss_fake = self._discriminator_step()
                        loss = self._generator_step()
                        self.D_limit *= self.D_limit_update
                        self.G_limit *= self.G_limit_update
                        step = "DG"
                        turns += 1
                    self.history["discriminator_loss_real"].append(discriminator_loss_real)
                    self.history["discriminator_loss_fake"].append(discriminator_loss_fake)
                    self.history["loss"].append(loss)
                    if verbose:
                        pb.step("{}({}), {}, {}, {}".format(step, turns, discriminator_loss_real, discriminator_loss_fake, loss))
            else:
                for i in range(iterations):
                    discriminator_loss_real, discriminator_loss_fake = self._discriminator_step()
                    loss = self._generator_step()
                    self.history["discriminator_loss_real"].append(discriminator_loss_real)
                    self.history["discriminator_loss_fake"].append(discriminator_loss_fake)
                    self.history["loss"].append(loss)
                    if verbose:
                        pb.step("{}, {}, {}".format(discriminator_loss_real, discriminator_loss_fake, loss))
            logging.info("Testing...")
            K.set_learning_phase(0)
            if type(self.test_data) != tuple:
                self.test_data = [next(self.test_data) for i in range(len(self.test_data))]
                for name, measure in self.measures.items():
                    mean = []
                    for data_item in self.test_data:
                        mean.append(measure(*data_item)) # what kinds of are required?
                    self.results[name] = np.array(mean).mean()
            else:
                for name, measure in self.measures.items():
                    self.results[name] = measure(*self.test_data) # what kinds of are required?
        finally:
            self._save()

    def _discriminator_step(self):
        ONES = 0.75 + 0.25 * np.random.random([self.batch_size, 1])
        ZEROS = 0.25 * np.random.random([self.batch_size, 1])
        t = time.time()
        lres, hres = next(self.train_data)
        fakes = self.model.predict(lres)
        if self.trainer.uses_learning_phase and not isinstance(K.learning_phase(), int):
            discriminator_loss_real = self.discriminator.train_function(
                [lres, hres, ONES, self.ONESFLAT, 1.])
            discriminator_loss_fake = self.discriminator.train_function(
                [lres, fakes, ZEROS, self.ONESFLAT, 1.])
        else:
            discriminator_loss_real = self.discriminator.train_function(
                [lres, hres, ONES, self.ONESFLAT])
            discriminator_loss_fake = self.discriminator.train_function(
                [lres, fakes, ZEROS, self.ONESFLAT])
        logging.log(11, "Training discriminator:", time.time() - t)
        return discriminator_loss_real, discriminator_loss_fake

    def _generator_step(self):
        t = time.time()
        lres, hres = next(self.train_data)
        if self.trainer.uses_learning_phase and not isinstance(K.learning_phase(), int):
            loss = self.trainer.train_function([lres, hres, self.ONES, hres, self.ONESFLAT, self.ONESFLAT, 1.])
        else:
            loss = self.trainer.train_function([lres, hres, self.ONES, hres, self.ONESFLAT, self.ONESFLAT])
        logging.log(11, "Training generator:", time.time() - t)
        return loss

    def _save(self):
        super()._save()
        self.discriminator.save(os.path.join(self.name, "discriminator.h5"))


# Generative Adversarial Networks

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

# https://github.com/soumith/ganhacks
# - don't use ReLUs, normalize to [-1; 1]
# - Dropout on Generation?
# the discriminator has to not overfit in order for GAN to work well
# -> add some noise to training - wrong labels
