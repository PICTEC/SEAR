from loaders.variable import FFTVariable

import numpy as np

class Feature:
    """
    Interface for classes generating feature extraction code, composable one feature at a time
    - preparation
    - size
    - code
    """

    _variables = []

    def __init__(self):
        self._size = None

    def preparation(self, sample_input):
        raise NotImplementedError

    @property
    def size(self):
        return self._size

    def code(self, beginning):
        raise NotImplementedError

    @property
    def variables(self):
        return self._variables[:]


class LogPowerRFFT(Feature):

    _variables = [FFTVariable("fft")]

    def preparation(self, sample_input):
        self._size = np.fft.rfft(sample_input).shape[0]
        return "pass"

    def code(self, beginning):
        return "features[{}:{}] = -np.log(np.abs(fft) ** 2 + 1e-12)".format(beginning, beginning + self._size)

class LogPowerRFFTWithoutZero(Feature):

    _variables = [FFTVariable("fft")]

    def preparation(self, sample_input):
        self._size = np.fft.rfft(sample_input).shape[0] - 1
        return "pass"

    def code(self, beginning):
        return "features[{}:{}] = -np.log(np.abs(fft[1:]) ** 2 + 1e-12)".format(beginning, beginning + self._size)


class NormalizedLogPowerRFFT(Feature):

    _variables = [FFTVariable("fft")]

    def preparation(self, sample_input):
        self._size = np.fft.rfft(sample_input).shape[0]
        return "pass"

    def code(self, beginning):
        return "features[{}:{}] = -np.log(np.abs(fft  / len(signal)) ** 2 + 1e-12)".format(beginning, beginning + self._size)


class NormalizedLogPowerRFFTWithoutZero(Feature):

    _variables = [FFTVariable("fft")]

    def preparation(self, sample_input):
        self._size = np.fft.rfft(sample_input).shape[0] - 1
        return "pass"

    def code(self, beginning):
        return "features[{}:{}] = -np.log(np.abs(fft[1:] / len(signal)) ** 2 + 1e-12)".format(beginning, beginning + self._size)


class PhaseRFFT(Feature):

    _variables = [FFTVariable("fft")]

    def preparation(self, sample_input):
        self._size = np.fft.rfft(sample_input).shape[0]
        return "pass"

    def code(self, beginning):
        return "features[{}:{}] = np.angle(fft)".format(beginning, beginning + self._size)



class Scale(Feature):

    def __init__(self, feature, scaler):
        self.feature = feature
        self.scaler = scaler
    
    @property
    def size(self):
        return self.feature.size
    
    @property
    def variables(self):
        return self.feature.variables
    
    def preparation(self, sample_input):
        return self.feature.preparation(sample_input)

    def code(self, beginning):
        code = self.feature.code(beginning)
        code += "\n\tfeatures[{}:{}] = features[{}:{}] * {}".format(beginning, beginning + self.feature._size,
            beginning, beginning + self.feature._size, self.scaler)
        return code


class Trim(Feature):

    def __init__(self, feature, trimsize):
        self.feature = feature
        self.trimsize = trimsize
    
    @property
    def size(self):
        return self.trimsize
    
    @property
    def variables(self):
        return self.feature.variables
    
    def preparation(self, sample_input):
        return self.feature.preparation(sample_input)

    def code(self, beginning):
        code = self.feature.code(beginning)
        code = code.split("\n\t")
        code = "\n\t".join(code[:-1]) + "\n\tfeatures[{}:{}] = ({})[:{}]".format(beginning, beginning + self.size,
            code[-1].split("=")[1], self.size)
        return code

def cqt(x, sr=16000, per_octave = 60, ):
    base_fq = float(sr) / len(x)
    octaves = np.log2((sr/2) / base_fq)
    n_bins = np.floor(octaves * per_octave).astype(np.int32)
    step = 2 ** (1. / per_octave)
    Q = (step ** 0.5) # half semitone(if per_octave = 12) in each direction
    X = np.zeros(n_bins, dtype=np.complex64)
    N = Q * (sr / 2) / (base_fq * step ** np.arange(n_bins))
    for k in range(n_bins):
        X[k] = np.sum(x * np.exp(-2 * 1j * np.pi * Q / N[k] * np.arange(len(x)))) /  N[k]
    return X

def icqt(spec_window, sr=16000, ):
    pass

