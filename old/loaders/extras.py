import GPUtil
import keras
import keras.backend as K
import numpy as xp
import os

def get_n_fft(magnitude):
    return magnitude.shape[-1] * 2 - 2

def get_stft(n_fft):
    n_hop = n_fft // 4
    window = xp.hanning(n_fft)
    spec_width = n_fft // 2 + 1
    def stft(signal):
        length = 1 + (signal.shape[0] - n_fft) // n_hop
        spec = xp.zeros([length, spec_width], dtype=xp.complex)
        for index in range(length):
            spec[index, :] = xp.fft.rfft(window * signal[
                index*n_hop : index*n_hop + n_fft
            ])
        return spec
    return stft

def get_istft(n_fft):
    n_hop = n_fft // 4
    def istft(spectro):
        length = n_fft + (spectro.shape[0] - 1) * n_hop
        signal = xp.zeros(length)
        for index in range(spectro.shape[0]):
            signal[index*n_hop : index*n_hop + n_fft] += xp.fft.irfft(spectro[index,:])
        return signal
    return istft

def delog_griffin_lim(magnitude, iterations=250, verbose=False):
    """
    Also normalizes...
    """
    magnitude = -1 * magnitude
    magnitude = xp.exp(magnitude) * get_n_fft(magnitude)
    return griffin_lim(magnitude, iterations, verbose)

def griffin_lim(magnitude, iterations=250, verbose=False):
    n_fft = get_n_fft(magnitude)  # check whether static offset is present
    n_hop = n_fft // 4
    signal_length = (magnitude.shape[0] - 1) * n_hop + n_fft
    signal = xp.random.random(signal_length)
    stft = get_stft(n_fft)
    istft = get_istft(n_fft)
    for iteration in range(iterations):
        reconstruction = stft(signal)
        phase = xp.angle(reconstruction)
        proposed = magnitude * xp.exp(1j * phase)
        prev_signal = signal
        signal = istft(proposed)
        RMSE = xp.sqrt(((prev_signal - signal)**2).mean())
        if verbose:
            print("Iteration {}/{} RMSE: {}".format(iteration + 1, iterations, RMSE))
    return signal

class AllPassKernelInitializer(keras.initializers.Initializer):
    def __init__(self, is_gru = False, value = 5.):
        self.is_gru = is_gru
        self.value = value

    def __call__(self, shape, dtype=None):
        if self.is_gru:
            return NotImplemented
        else:
            step = int(shape[0] / 4)
            width = shape[1]
            print(step, width)
            assert step == width
            return K.concatenate([K.zeros((step, width), dtype = dtype),
                K.constant(self.value, shape = K.eye(step), dtype = dtype),
                K.zeros((step, width), dtype = dtype),
                K.zeros((step, width), dtype = dtype)])


class LossHistory(keras.callbacks.Callback): # move it outta here
    def __init__(self):
        super().__init__()
        self.max_error = []
        self.mean_error = []
    def on_batch_end(self, batch, logs={}):
        self.max_error.append(logs.get('maximal_error'))
        self.mean_error.append(logs.get('mean_absolute_error'))
    def on_train_end(self, logs):
        import json
        print(self.max_error)
        print(self.mean_error)
        self.max_error = [float(x) for x in self.max_error]
        self.mean_error = [float(x) for x in self.mean_error]
        with open("logs.json", "w") as f:
            f.write(json.dumps({
                "max_error": self.max_error,
                "mae": self.mean_error
            }))


def set_device():
    try:
        GPUtil.getFirstAvailable(attempts=3, interval=0.5)
    except (RuntimeError, ValueError):
        print("Switching to CPU only as GPU is busy or unavailable")
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def add_coord_2d(layer):
    shape = layer.shape[1:].as_list()
    x = Lambda(lambda x: 
       	K.stack([K.tile(K.stack([K.arange(shape[0])]), shape[1])], axis=-1))(layer)
    y = Lambda(lambda x:
       	K.stack([K.tile(K.stack([K.arange(shape[0])], axis=-1), shape[1])], axis=-1))(layer)
    return concatenate([layer, x, y])
