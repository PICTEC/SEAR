import datetime
import dill
import gc
import GPUtil
import json
import numpy as np
import os
import random
import re
import requests
import scipy.io.wavfile as sio
import subprocess
import tempfile
import zipfile

import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Lambda, LeakyReLU, TimeDistributed, Flatten, BatchNormalization
import keras.backend as K
import keras.callbacks as kc
import tensorflow as tf


class ShowMeta(type):
    def __str__(cls):
        return "\n".join(["{}: {}".format(k, v) for k, v in dict(vars(cls)).items() if not k.startswith("_")])

    def __repr__(cls):
        return "\n".join(["{}: {}".format(k, v) for k, v in dict(vars(cls)).items() if not k.startswith("_")])

    def __format__(cls):
        return "\n".join(["{}: {}".format(k, v) for k, v in dict(vars(cls)).items() if not k.startswith("_")])

    
def set_device():
    try:
        GPUtil.getFirstAvailable(attempts=3, interval=0.5)
    except (RuntimeError, ValueError):
        print("Switching to CPU only as GPU is busy or unavailable")
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

def pesq(gt, pred, phase):
    spec = (np.sqrt(np.exp(-gt)) * 512) * np.exp(phase * 1j)
    sound = np.zeros(spec.shape[0] * 128 + 512 - 128)
    for i in range(spec.shape[0]):
        frame = np.fft.irfft(spec[i,:])
        sound[128 * i : 128 * i + 512] += frame
    spec = (np.sqrt(np.exp(-pred)) * 512) * np.exp(phase * 1j)
    sound2 = np.zeros(spec.shape[0] * 128 + 512 - 128)
    for i in range(spec.shape[0]):
        frame = np.fft.irfft(spec[i,:])
        sound2[128 * i : 128 * i + 512] += frame
    fname_gt = tempfile.mktemp() + ".wav"
    fname_pred = tempfile.mktemp() + ".wav"
    # print(sound.shape, sound2.shape)
    sio.write(fname_gt, 16000, (2**15 * sound).astype(np.int16))
    sio.write(fname_pred, 16000, (2**15 * sound2).astype(np.int16))
    ot, e = subprocess.Popen(["PESQ", "+wb", "+16000", fname_gt, fname_pred], stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()
    os.remove(fname_gt)
    os.remove(fname_pred)
    # print(ot)
    o = ot.decode("utf-8").split('\n')[-2]
    # print(o, len(o))
    # if not len(o):
    #     print(ot.decode("utf-8"))
    value = re.findall("= \d\.\d+", o)[0]
    # print(value)
    return float(value[2:])


# log-spectral distance

def LSD(gt, pred):
    innermost = (10 * ((-pred) - (-gt)) / np.log(10)) ** 2
    for i in range(gt.shape[0]):
        inner = innermost[i, :, :]
        length = len(np.where((gt[i, :, :] != 0).sum(1))[0])
        inner = inner[:length]
        sublsd = []
        for t in range(length):
            step = 2 / 513
            frame = inner[t]
            integral = frame.sum()
            sublsd.append(np.sqrt(step * integral))
    return np.array(sublsd)


class Config(metaclass=ShowMeta):
    
    ROOT = os.path.expanduser("~")
    PATH = os.path.join(ROOT, "DAE-libri")
    CACHE_PATH = os.path.join(ROOT, "cache")
    DATASETPATH = os.path.join(ROOT, ".dataset")
    NOISEPATH = os.path.join(ROOT, "new_NoiseDb")
    MODEL_ROOT = os.path.join(ROOT, "Denoising/runs")

    @staticmethod
    def set_gpu():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        
class Task:
    """
    Create a probeable task for ML
    """


class DatasetPreparation(Task):
    
    @staticmethod
    def default():
        TRAIN = 160 # in paper - 9k
        VALID = 80 # in paper - 80
        TEST  = 80 # in paper - 80
        return DatasetPreparation(TRAIN, VALID, TEST)

    @staticmethod
    def setting_for(codec_type, SNR, training_data=9000, mask=False):
        TRAIN = training_data # 9000
        VALID = 80 # 80
        TEST  = 80 # 80
        do_i_noise = not SNR is None
        return DatasetPreparation(TRAIN, VALID, TEST, codec_type=codec_type, do_i_noise=do_i_noise, SNR=SNR, mask=mask)

    
    def __init__(self, TRAIN, VALID, TEST, codec_type=None, do_i_noise=True, SNR=5, mask=False):
        self.SNR = SNR
        self.do_i_noise = do_i_noise
        self.TRAIN = TRAIN
        self.VALID = VALID
        self.TEST = TEST
        files = [os.path.join(Config.PATH, x) for x in os.listdir(Config.PATH) if x.endswith(".wav") and not x.endswith(".gsm.wav")]
        self.train_files = random.sample(files, TRAIN)
        self.valid_files = random.sample(list(set(files) - set(self.train_files)), VALID)
        self.test_files = random.sample(list(set(files) - set(self.train_files) - set(self.valid_files)), TEST)
        self.noise_files = [os.path.join(Config.NOISEPATH, x) for x in os.listdir(Config.NOISEPATH)]
        self.LENGTH = None
        self.codec_type = codec_type if codec_type else "amr-lq"
        self.mask = mask
        assert self.codec_type in ("amr-lq", "amr-hq", "gsm-fr")
        
    def run(self):
        self.LENGTH = 1248
        SOURCE_SHAPE = [self.LENGTH, 129]
        TARGET_SHAPE = [self.LENGTH, 257]
        [self._prepare_data(x) for x in self.train_files]
        [self._prepare_data(x) for x in self.valid_files]
        [self._prepare_data(x) for x in self.test_files]
        train_source = np.zeros([self.TRAIN] + SOURCE_SHAPE, np.float32)
        valid_source = np.zeros([self.VALID] + SOURCE_SHAPE, np.float32)
        test_source  = np.zeros([self.TEST]  + SOURCE_SHAPE, np.float32)
        train_target = np.zeros([self.TRAIN] + TARGET_SHAPE, np.float32)
        valid_target = np.zeros([self.VALID] + TARGET_SHAPE, np.float32)
        test_target  = np.zeros([self.TEST]  + TARGET_SHAPE, np.float32)
        test_phase   = np.zeros([self.TEST]  + TARGET_SHAPE, np.float32)    
        [self._get_data(train_source, train_target, ix, x) for ix, x in enumerate(self.train_files)]
        [self._get_data(valid_source, valid_target, ix, x) for ix, x in enumerate(self.valid_files)]
        [self._get_data(test_source,  test_target,  ix, x, true_phase=test_phase) for ix, x in enumerate(self.test_files)]
        if not os.path.exists(Config.DATASETPATH):
            os.mkdir(Config.DATASETPATH)
        if self.mask:
            train_target = np.clip(np.nan_to_num(train_target / (train_source + 2e-12)), 0, 10)
            valid_target = np.clip(np.nan_to_num(valid_target / (valid_source + 2e-12)), 0, 10)
        np.save(os.path.join(Config.DATASETPATH, "train_source.bin"), train_source)
        np.save(os.path.join(Config.DATASETPATH, "valid_source.bin"), valid_source)
        np.save(os.path.join(Config.DATASETPATH, "test_source.bin"), test_source)
        np.save(os.path.join(Config.DATASETPATH, "train_target.bin"), train_target)
        np.save(os.path.join(Config.DATASETPATH, "valid_target.bin"), valid_target)
        np.save(os.path.join(Config.DATASETPATH, "test_target.bin"), test_target)
        np.save(os.path.join(Config.DATASETPATH, "test_phase.bin"), test_phase)
        gc.collect()

    _tempnam = tempfile.mktemp
        
    def _ennoise(self, data):
        SNR_ln = self.SNR / 10 * np.log(10)
        noise = sio.read(random.choice(self.noise_files))[1].astype(np.float32)
        data = data[:176000].astype(np.float32)
        if len(noise) <= len(data):
            print("NOISE TOO SHORT:", len(noise))
            noise = np.pad(noise, ((0, len(data) - len(noise)),),'constant')
        else:
            start = random.randint(0, len(noise) - len(data) - 1)
            noise = noise[start:start + len(data)]
        log_power_of_signal = np.log((data ** 2).mean() + 2e-12)
        log_power_of_noise = np.log((noise ** 2).mean() + 2e-12)
        gain_of_noise = np.exp(log_power_of_signal - SNR_ln - log_power_of_noise)   # SNR was miscalculated...
        gain_of_noise = np.random.uniform(0, gain_of_noise)
        if np.isnan(gain_of_noise):
            print("NAN gain - generating some random white noise")
            noise = np.random.normal(0, 1500, size=data.shape)
        else:
            noise = noise * gain_of_noise
        noised = data + noise
        if np.abs(noised).max() >= 2**15 - 1:
            print("Clipping noised by", np.abs(noised).max() - 2**15 - 1)
            noised = np.clip(noised, -1 * 2**15, 2**15 - 1)
        noised = noised.astype(np.int16)
        # print(gain_of_noise, log_power_of_signal, log_power_of_noise)
        # print("Mean amp of data: ", np.abs(data).mean())
        # print("Mean amp of noise: ", np.abs(noise).mean())
        # print("Mean amp of noised signal: ", np.abs(noised).mean())
        return noised

    def _prepare_data(self, filename):
        print(filename)
        data = sio.read(filename)[1] # .astype(np.float32) to generate noise in the experiment...
        if self.do_i_noise:
            data = self._ennoise(data)
        oldname = DatasetPreparation._tempnam() + '.oldwav'
        sio.write(oldname, 16000, data)
        newname = filename + ".gsm.wav"
        if self.codec_type == "amr-hq":
            tmpname = DatasetPreparation._tempnam() + '.amr-nb'
            subprocess.Popen(['sox', oldname, '-C', '7', '-r', '8000', tmpname]).communicate()
        elif self.codec_type == "amr-lq":
            tmpname = DatasetPreparation._tempnam() + '.amr-nb'
            subprocess.Popen(['sox', oldname, '-C', '0', '-r', '8000', tmpname]).communicate()
        elif self.codec_type == "gsm-fr":
            tmpname = DatasetPreparation._tempnam() + '.gsm'
            subprocess.Popen(['sox', oldname, '-r', '8000', tmpname]).communicate()
        subprocess.Popen(['sox', tmpname, '-r', '16000', "-e", "signed", '-b', '16',  newname]).communicate()
        list(map(os.remove, [oldname, tmpname]))

    _window = np.hamming(512)

    def _get_data(self, source, target, index, filename, true_phase=None):
        print(filename)
        recording = sio.read(filename + ".gsm.wav")[1].astype(np.float32)
        recording /= 2**15
        for time in range(self.LENGTH):
            win = recording[128 * time : 128 * time + 512]
            if len(win) != 512:
                break
            fft = np.fft.rfft(DatasetPreparation._window * win) / 512
            source[index, time, :] = -np.log(np.abs(fft) ** 2 + 2e-12)[:129]
        recording = sio.read(filename)[1].astype(np.float32)
        recording /= 2**15
        for time in range(self.LENGTH):
            win = recording[128 * time : 128 * time + 512]
            if len(win) != 512:
                break
            fft = np.fft.rfft(DatasetPreparation._window * win) / 512
            target[index, time, :] = -np.log(np.abs(fft) ** 2 + 2e-12)
            if true_phase is not None:
                true_phase[index, time, :] = np.angle(fft)

                
class StopOnConvergence(kc.Callback):
    """Callback that terminates training when a NaN loss is encountered.
    """

    def __init__(self, max_repetitions=10):
        super().__init__()
        self.max_repetitions = max_repetitions
    
    def on_train_begin(self, logs=None):
        self.repetitions = 0
        self.last_loss = np.inf
    
    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('val_loss')
        if loss is not None:
            if loss > self.last_loss:
                self.repetitions += 1
            else:
                self.last_loss = loss
                self.repetitions = 0
            if self.repetitions > self.max_repetitions:
                self.model.stop_training = True

                
class SlackNotifications:
    def __init__(self, token=None):
        if token is not None:
            self.token = token
        elif os.path.isfile(".slack.token"):
            self.token = open(".slack.token").read().strip()
        elif os.path.isfile(os.expanduser("~/.local/.slack.token")):
            self.token = open(os.expanduser("~/.local/.slack.token")).read()
        else:
            raise ValueError("Token not present and cannot be found in config files")
        
    def notify(self, message):
        headers = {
            'Content-type': 'application/json',
        }
        data = json.dumps({"text": message})
        return requests.post(
            self.token,
            headers=headers, data=data)
        

class Training(Task):
    def __init__(self, model_schema, name=None, notifier=None, loss=None):
        self.model_schema = model_schema
        self.model_name = name if name is not None else "model"
        self.notifier = notifier
        self.loss = loss if loss is not None else 'mse'
            
    def run(self):
        Config.set_gpu()
        train_source = np.load(os.path.join(Config.DATASETPATH, "train_source.bin.npy"))
        valid_source = np.load(os.path.join(Config.DATASETPATH, "valid_source.bin.npy"))
        train_target = np.load(os.path.join(Config.DATASETPATH, "train_target.bin.npy"))
        valid_target = np.load(os.path.join(Config.DATASETPATH, "valid_target.bin.npy"))
        MAX = np.abs(train_source).max()
        model = self.model_schema.make(MAX=MAX)
        name = datetime.datetime.now().isoformat() + ".zip"
        try:
            for lr in [3e-4, 1e-4, 3e-5, 1e-5]:
                model.model.compile(keras.optimizers.Adam(lr), self.loss)
                model.model.fit(train_source, train_target, validation_data=(valid_source, valid_target), batch_size=8, epochs=100, callbacks=[kc.TensorBoard("./tf_logs"), StopOnConvergence(4)])
        finally:
            model.name = self.model_name + "_" + name
            model.save(os.path.join(Config.MODEL_ROOT, name))
            print("Model path: ", os.path.join(Config.MODEL_ROOT, model.name))
            if self.notifier is not None:
                self.notifier.notify("Training of {} has ended".format(model.name))
        return model


class Evaluation(Task):
    def __init__(self, model):
        self.model = model
        
    def run(self, print_results=True):
        test_source = np.load(os.path.join(Config.DATASETPATH, "test_source.bin.npy"))
        test_target = np.load(os.path.join(Config.DATASETPATH, "test_target.bin.npy"))
        test_phase =  np.load(os.path.join(Config.DATASETPATH, "test_phase.bin.npy"))
        preds = self.model.model.predict(test_source)
        mse = np.sqrt(((preds - test_target) ** 2).mean((1,2)))
        mae = np.abs(preds - test_target).mean((1,2))
        pesq = self.pesq_loop(test_target, preds, test_phase)
        lsd = LSD(test_target, preds)
        if print_results:
            print(f"Model name: {self.model.name}, MSE: {mse.mean()} +- {mse.std()}, MAE: {mae.mean()} +- {mae.std()}, PESQ: {pesq.mean()} +- {pesq.std()}, LSD: {lsd.mean()} +- {lsd.std()}")
        return {"name": self.model.name, "MSE": mse, "MAE": mae, "PESQ": pesq, "LSD": lsd}

    def pesq_loop(self, gt, preds, phase):
        quality = []
        for i in range(gt.shape[0]):
            try:
                quality.append(pesq(gt[i], preds[i], phase[i]))
            except IndexError:
                print("Failed getting PESQ value for recording {}".format(i))
        return np.array(quality)
    
    
class ModelInstance:
    """
    Should extend with save and integrations
    TODO: convert batch_size(as extension)
    """
    
    _imports = {"tf": tf}
    
    def __init__(self, model, meta=None):
        self.model = model
        self.meta = meta

    def save(self, path):
        tdir = tempfile.TemporaryDirectory()
        if self.meta:
            dill.dump(self.meta, open(os.path.join(tdir.name, "meta.pkl"), "wb"))
        keras.models.save_model(self.model, os.path.join(tdir.name, "model.h5"))
        zip_file = zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED)
        for root, _, files in os.walk(tdir.name):
            for file in files:
                zip_file.write(os.path.join(root, file), file)        
        [print(zinfo) for zinfo in zip_file.filelist]
        zip_file.close()
        
    @classmethod
    def load(cls, path):
        tdir = tempfile.TemporaryDirectory()
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(tdir.name)
        zip_ref.close()
        if os.path.isfile(os.path.join(tdir.name, "meta.pkl")):
            meta = dill.load(open(os.path.join(tdir.name, "meta.pkl"), "rb"))
            meta.update(cls._imports)
            mdl = keras.models.load_model(os.path.join(tdir.name, "model.h5"), meta)
        else:
            meta = None
            mdl = keras.models.load_model(os.path.join(tdir.name, "model.h5"), cls._imports)
        return ModelInstance(mdl, meta)
        


class ModelSchema:
    """
    What is the point of schema?
    """
    def __init__(self, cb):
        self._cb = cb
        
    def make(self, **kwargs):
        model = self._cb(**kwargs)
        return ModelInstance(model, meta=kwargs)


try:
    import tensorflow.signal as tfsignal
    import tensorflow.spectral as tfspectral
except ImportError:
    import tensorflow.contrib.signal as tfsignal
    import tensorflow.spectral as tfspectral


class LossSum:
    def __init__(self, *components):
        self.components = components
        self.__name__ = self.__class__.__name__

    def __call__(self, true, preds):
        return sum([comp(true, preds) for comp in self.components])


class MFCCLossComponent:
    def __init__(self, rate=0.1, depth=26, n_fft=257, sample_rate=16000, bounds=(50, 8000)):
        self.depth = depth
        self.rate = K.cast_to_floatx(rate)
        self.__name__ = self.__class__.__name__
        self.weights = tfsignal.linear_to_mel_weight_matrix(
            num_mel_bins=depth,
            num_spectrogram_bins=n_fft,
            sample_rate=sample_rate,
            lower_edge_hertz=bounds[0],
            upper_edge_hertz=bounds[1]
        )

    def __call__(self, true, preds):
        mfcc_true = tfspectral.dct(K.dot(true, self.weights))
        mfcc_preds = tfspectral.dct(K.dot(preds, self.weights))
        return self.rate * K.mean((mfcc_true - mfcc_preds)**2)



if __name__ == "__main__":
    dset_task = DatasetPreparation.default()
    dset_task.run()
    train_task = Training(ModelSchema(mk_model))
    model = train_task.run()
    eval_task = Evaluation(model)
    results = eval_task.run(print_results=True)
