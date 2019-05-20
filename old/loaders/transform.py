import abc
import librosa
import numpy as np
import os
import random
import scipy.io.wavfile as sio
import subprocess
import tempfile
import types

from loaders.dataset import manager

# replacement for os.tempnam
def tempnam():
    f = tempfile.NamedTemporaryFile(delete = False, dir = "cache") # TODO cache parametrizable
    f.close()
    os.remove(f.name)
    return f.name


class Transform(metaclass=abc.ABCMeta):
    """
    Base class for all feature extractors
    """
    @property
    def source(self):
        """
        "signal" - get loaded signal
        "fname" - name of the file
        "fpath" - load path of the file
        "feature:<index/name>" - get from metadata
        Supports caching...
        """
        return self._source

    def make_sources(self):
        return {}

    @abc.abstractmethod
    def __call__(self, input):
        raise NotImplementedError()

    def inverse(self, output):
        if hasattr(self, "_inverse"):
            return self._inverse(output)
        raise NotImplementedError("Transform inverse not specified: {}".format(self.__class__))
                                
    def add_inverse(self, fun):
        """
        Takes a function and makes it an inverse of the transform
        Overrides the original transform
        """
        self._inverse = types.MethodType(lambda x,y: fun(y), self)


class TransformModifier(Transform):
    """
    Destructive by default, hence inverse is not productive
    You must specify inversion manually if you want to
    """

    def __call__(self, input):
        return self.transform(input)

    def make_sources(self):
        sources = {self.source: self.outer}
        sources.update(self.transform.make_sources())
        return sources

    def inverse(self, output):
        if hasattr(self, "_inverse"):
            return self._inverse(output)
        return self.transform.inverse(output)

    def add_inverse(self, fun):
        """
        Takes a function and makes it an inverse of the transform
        Overrides the original transform
        """
        self._inverse = types.MethodType(lambda x,y: fun(y), self)


class FeatureTransform(Transform):

    _source = "signal"

    def __init__(self):
        self._compiled = False
        self._dtype = np.float32
        self.features = []

    def add_feature(self, feature):
        self.features.append(feature)
        self._compiled = False

    def _code_join(self, statements, init=False):
        if init:
            return "def prep_fn(self):\n\t" + "\n\t".join(statements) + "\n"
        return "def fn(self, signal):\n\tfeatures = np.zeros(" +\
            str(self._feature_count) + ", dtype=np.float32)\n\t" + "\n\t".join(statements) + "\n\treturn features\n"

    def _compile(self, sample_input):
        """
        Generate method by joining various feature classes
        - _extraction_function(self, input)
        - _feature_count
        """
        preparation_code = self._code_join(
            [feat.preparation(sample_input) for feat in self.features], init=True)
        l = {}
        exec(preparation_code, globals(), l)
        prep_fn = l['prep_fn']
        self._extraction_init = types.MethodType(prep_fn, self)
        # what with variables
        variables = [item for feature in self.features for item in feature.variables]
        unique = []
        [unique.append(x) for x in variables if x.key not in [i.key for i in unique]]
        variables = unique[:]
        sizes = [feat.size for feat in self.features]
        sizes = np.cumsum(np.array([0] + sizes)).tolist()
        self._feature_count = sizes.pop()
        function_code = self._code_join([var.code for var in variables] +
                    [feat.code(beginning) for beginning, feat in zip(sizes,self.features)])
        l = {}
        exec(function_code, globals(), l)
        fn = l['fn']
        self._extraction_function = types.MethodType(fn, self)


class Windowing(FeatureTransform):
    def __init__(self, trim_size=0):
        FeatureTransform.__init__(self)
        self._window_length = 512
        self._window_slide = 128
        self._window_function = np.hamming
        self._trim_size = trim_size

    @property
    def window_length(self):
        return self._window_length

    @window_length.setter
    def set_window_length(self, val):
        self._compiled = False
        self._window_length = val

    @property
    def window_slide(self):
        return self._window_slide

    @window_slide.setter
    def set_window_slide(self, val):
        self._compiled = False
        self._window_slide = val

    @property
    def window_function(self):
        return self._window_function

    @window_function.setter
    def set_window_function(self, val):
        self._compiled = False
        self._window_function = val

    def __call__(self, signal):
        if not self._compiled:
            self._compile(signal[:self.window_length])
            self._compiled = True
        length = 1 + (signal.shape[0] - self.window_length) // self.window_slide
        win_mask = self.window_function(self.window_length)
        features = np.zeros((length - self._trim_size, self._feature_count), self._dtype)
        # print(features.shape)
        self._extraction_init()
        for index in range(length - self._trim_size):
            # TODO: add buffering
            # TODO: abstract to step extractor
            window = signal[index * self.window_slide : index * self.window_slide + self.window_length]
            features[index,:] = self._extraction_function(win_mask * window)
        return features

    def detransformer(self):
        # TODO: this is a mockup, detransformer should be built for each transform if possible ;)
        def detransform(data):
            n_time = data.shape[0]
            n_freq = data.shape[1]
            recording = np.zeros([(n_time - 1) * self.window_slide + self.window_length], dtype=np.float32)
            for time in range(n_time):
                feats = np.sqrt((np.concatenate([np.zeros(1, dtype=np.float32), np.exp(-data[time, :])])))
                recording[time * self.window_slide : time * self.window_slide + self.window_length] += np.fft.irfft(feats)
            return recording
        return detransform


class AddGaussianNoise(TransformModifier):

    _source = "extra:gaussian_noise"

    def __init__(self, transform, variance=0.1):
        assert issubclass(transform.__class__, Transform)
        self.transform = transform
        self.variance = variance

    def outer(self, dataset):
        def inner(f):
            data = dataset._sources(self.transform.source)(f) # TODO: not use private
            data += np.random.normal(0, self.variance, size=data.shape)
            return data
        return inner


class AddLambdaNoise(TransformModifier):

    _counter = 0

    def __init__(self, transform, noise_lambda):
        self._counter += 1
        self.index = self._counter
        self.transform = transform
        self.noise_lambda = noise_lambda

    def outer(self, dataset):
        def inner(f):
            data = dataset._sources(self.transform.source)(f) # TODO: not use private
            data += noise_lambda(data.shape)
            return data
        return inner

    @property
    def source(self):
        return "extra:lambda_noise:" + str(self.index)


class MixNoise(TransformModifier):

    _counter = 0

    def __init__(self, transform, noise_source, PPSNR = 10):
        self._counter += 1
        self.index = self._counter
        self.transform = transform
        self.noise_sources = [os.path.join(noise_source, x) for x in os.listdir(noise_source)]
        self.PPSNR = PPSNR # peak-signal peak-noise power ratio

    def outer(self, dataset):
        def inner(f):
            # TODO: prepare for mono
            data = dataset._sources(self.transform.source)(f) # TODO: not use private
            noise = sio.read(random.choice(self.noise_sources))[1].astype(np.float32) / 2**15
            if len(noise) - len(data) - 1 > 0:
                selection = random.randint(0, len(noise) - len(data) - 1)
                noise = noise[selection : selection + len(data)]
            else:
                noise_data = np.zeros(len(data))
                noise_data[:len(noise)] = noise
                noise = noise_data
            proper_ratio = 10 ** (self.PPSNR / 10.)
            max_ratio = (data ** 2).max() / (noise ** 2).max()
            noise_gain = np.sqrt(max_ratio / proper_ratio)
            new_data = data + noise_gain * noise
            new_data *= np.abs(data).mean() / np.abs(new_data).mean()
            return new_data
        return inner

    @property
    def source(self):
        return "extra:mix_noise:" + str(self.index)


class MixReverb(TransformModifier):

    _counter = 0

    def __init__(self, transform, reverb_source):
        self._counter += 1
        self.index = self._counter
        self.transform = transform
        self.reverb_sources = [os.path.join(reverb_source, x) for x in os.listdir(reverb_source)]

    def outer(self, dataset):
        def inner(f):
            # TODO: prepare for mono
            data = dataset._sources(self.transform.source)(f) # TODO: not use private
            noise = sio.read(random.choice(self.reverb_sources))[1].astype(np.float32) / 2**15
            new_data = np.convolve(data, noise, mode='same')[:len(data)]
            data *= np.abs(data).mean() / np.abs(new_data).mean() # for proper scale of recordings
            return data
        return inner

    @property
    def source(self):
        return "extra:mix_reverb:" + str(self.index)


class GSMize(TransformModifier):

    _source = "extra:gsmized"

    def __init__(self, transform, packet_loss_rate = 0.,
            packet_loss_burst_range=None, bit_error_rate = 0.,
            packet_loss_concealment = 'zeros'):
        self.packet_loss_rate = packet_loss_rate
        self.packet_loss_burst_range = packet_loss_burst_range
        self.packet_loss_concealment = packet_loss_concealment
        self.bit_error_rate = bit_error_rate
        self.transform = transform

    def outer(self, dataset):
        def inner(fname):
            data = dataset._sources(self.transform.source)(fname) # TODO: not use private
            if self.packet_loss_rate > 0.:
                frames = int(np.ceil(len(data) / 160.))
                lost_frames = (np.random.random(frames) < self.packet_loss_rate).nonzero()[0].tolist()
                if self.packet_loss_burst_range is not None:
                    total_losses = []
                    for x in lost_frames:
                        burst = np.choice(range(self.packet_loss_burst_range[0],
                                                self.packet_loss_burst_range[1] + 1))
                        total_losses += range(x, x + burst)
                    lost_frames = [x for x in total_losses if x < frames]
                buffer = []
                for i in range(frames):
                    if i not in lost_frames:
                        buffer.append(data[160 * i : 160 * i + 160])
                data = np.concatenate(buffer)
            else:
                lost_frames = []
            oldname = tempnam() + '.oldwav'
            sio.write(oldname, 16000, data)
            tmpname = tempnam() + '.gsm'
            newname = tempnam() + '.wav'
            subprocess.Popen(['sox', oldname, '-r', '8000', tmpname]).communicate()
            subprocess.Popen(['sox', tmpname, '-r', '16000', "-e", "signed", '-b', '16',  newname]).communicate()
            data = sio.read(newname)[1].astype(np.float32) / 2**15  # maybe dataset._load abstraction?
            if lost_frames:
                buffer = []
                j = 0
                for i in range(frames):
                    if i not in lost_frames:
                        buffer.append(data[160 * (i - j) : 160 * (i - j) + 160])
                    else:
                        if self.packet_loss_concealment == 'zeros':
                            buffer.append(np.zeros(160, dtype=np.float32))
                        else:
                            raise NotImplementedError("Packet loss concealment method unavailable")
                        j += 1
                data = np.concatenate(buffer)
            list(map(os.remove, [oldname, tmpname, newname]))
            return data
        return inner


class ConstantQTransform(Transform):

    _source = "signal"

    def __init__(self, fmin=32.5, n_bins=84, bins_per_octave=12,  hop=128):
        self.cqt_callback = (lambda signal: 
            librosa.cqt(signal, fmin=fmin, n_bins=n_bins, hop_length=hop, bins_per_octave=bins_per_octave))

    def __call__(self, signal):
        return self.cqt_callback(signal).T


class Transcript(Transform):

    _source = "fpath"

    def __init__(self, split_char=" ", name_transform=(lambda input: ".".join(input.split(".")[:-1]) + ".txt"), verbose=True):
        self.split_char = split_char
        self.name_transform = name_transform
        self.symbols = set()
        self.translation = {} if manager is None else manager.dict()
        self.verbose = verbose
        self.len = 0

    def __call__(self, input):
        fname = self.name_transform(input)
        with open(fname) as f:
            return [self.translation[x] for x in f.read().strip().split(self.split_char)]

    def fit(self, names):
        if self.verbose:
            print("Fitting the transcript transform")
        steps = len(names) // 50
        for index, name in enumerate(names):
            name = name[:-4] + ".txt"
            with open(name) as f:
                tokens = f.read().strip().split(self.split_char)
                self.len = max(self.len, len(tokens))
                tokens = set(tokens)
                tokens = {x for x in tokens if x not in self.symbols}
                for token in tokens:
                    self.translation[token] = len(self.translation)
                self.symbols |= tokens
                if index % steps == 0:
                    if self.verbose:
                        print(".", end="")
        if self.verbose:
            print("")
            print("Total {} tokens".format(len(self.symbols)))
        return len(self.symbols), self.len


class LengthOfWindow(Transform):

    _source = "signal"

    def __init__(self, length, hop):
        self.length = length
        self.hop = hop

    def __call__(self, input):
        return (len(input) - self.length) // self.hop + 1


class Length(TransformModifier):

    def __init__(self, transform):
        self.transform = transform
        self._source = self.transform._source

    def make_sources(self):
        return self.transform.make_sources()

    def __call__(self, input):
        return len(self.transform(input))


class Lengthen(TransformModifier):
    def __init__(self, transform, desired_length, pad_with=0):
        self.transform = transform
        self.desired_length = desired_length
        self._source = self.transform._source
        self.pad_with = pad_with

    def make_sources(self):
        return self.transform.make_sources()
    
    def __call__(self, input):
        result = np.array(self.transform(input))
        desired = list(result.shape)
        desired[0] = self.desired_length
        desired = np.zeros(desired, result.dtype)
        desired[:] = self.pad_with
        desired[:len(result)] = result
        return desired


class DivisiblePad(TransformModifier):
    def __init__(self, transform, desired_length, pad_with=0):
        self.transform = transform
        self.desired_length = desired_length
        self._source = self.transform.source
        self.pad_with = pad_with

    def make_sources(self):
        return self.transform.make_sources()
    
    def __call__(self, input):
        result = np.array(self.transform(input))
        desired = list(result.shape)
        desired[0] += (self.desired_length - desired[0] % self.desired_length) if desired[0] % self.desired_length else self.desired_length
        desired = np.zeros(desired, result.dtype)
        desired[:] = self.pad_with
        desired[:len(result)] = result
        return desired


class Null(Transform):

    _source = "fname"

    def __call__(self, input):
        return 0
