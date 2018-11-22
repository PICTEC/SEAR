import keras
import numpy as np

EPSILON = 1e-38

from loaders.transform import Windowing

class Measure(object):
    """
    TODO: should take arguments regarding which inputs//outputs are taken as arguments against the measurement
    """
    def serialize(self):
        return self


class InnerMeasure(object):
    """
    Generally, a superclass of everything configured by Measures
    Returns parent constructor on serialize
    """
    def __init__(self, parent):
        self._parent = parent

    def serialize(self):
        return self._parent.serialize()


class MSE(Measure):
    def __init__(self, A=None, B=None):
        self.A = A
        self.B = B

    def __call__(self, model):
        A = self.A
        B = self.B
        class MSEInstance(InnerMeasure):
            def __call__(self, trainX, trainY):
                predictions = model.predict(trainX)
                if A is not None:
                    predictions = predictions[A]
                if B is not None:
                    trainY = trainY[B]
                return keras.losses.mse(trainY, predictions).eval(session=keras.backend.get_session()).mean()

        return MSEInstance(self)


class NullMSE(Measure):
    """
    If trainX and trainY have different shapes, unift them
    """

    def __init__(self, A=None, B=None):
        self.A = A
        self.B = B

    def __call__(self, model):
        A = self.A
        B = self.B
        class NullMSEInstance(InnerMeasure):
            def __call__(self, trainX, trainY):
                if A is not None:
                    trainX = trainX[A]
                if B is not None:
                    trainY = trainY[B]
                trainX = trainX[:, :, :min(trainX.shape[-1], trainY.shape[-1])]
                trainY = trainY[:, :, :min(trainX.shape[-1], trainY.shape[-1])]
                return keras.losses.mse(trainY, trainX).eval(session=keras.backend.get_session()).mean()
        return NullMSEInstance(self)


class LogSpectralDistance(Measure):

    def __init__(self, power=False):
        self.power = power

    def __call__(self, model):


        class LSDInstance(InnerMeasure):
            def __call__(self, trainX, trainY):
                predictions = model.predict(trainX)
                accumulator = 0.0
                for instance in range(predictions.shape[0]):
                    for time_frame in range(predictions.shape[1]):
                        accumulator += self._lsd(predictions[instance, time_frame, :],
                                trainY[instance, time_frame, :])
                return accumulator / (predictions.shape[0] * predictions.shape[1])

            @staticmethod
            def _lsd(X, Y):
                # based on https://en.wikipedia.org/wiki/Log-spectral_distance
                # inner = (10 * np.log10((X + EPSILON) / (Y + EPSILON)))**2
                inner = (10 * X / Y) ** 2  # since they're already log-powers
                step = np.pi / len(inner)
                inner = np.concatenate([np.array([0.0]), inner])
                integral = ((inner[:-1] + inner[1:]) * step / 2).sum() # trapeze method
                return np.sqrt(integral / np.pi) # integral is symmetric, so times 2, which cancels the denominator

        return LSDInstance(self)


class TimeDomainMSE(Measure):

    def __init__(self, windowing_transform, detransform='fft_no_zero', normalizer = None):
        self.windowing_transform = windowing_transform
        self.normalizer = normalizer
        self.detransform = detransform

    def __call__(self, model):

        detransformer = self.windowing_transform.detransformer() # HACK: to be replaced with correct detransformer

        class TimeMSEInstance(InnerMeasure):
            def __call__(self, trainX, trainY):
                predictions = model.predict(trainX)
                predictions = np.stack([detransformer(predictions[ex, :, :]) for ex in range(predictions.shape[0])])
                trainY = np.stack([detransformer(trainY[ex, :, :]) for ex in range(trainY.shape[0])])
                return keras.losses.mse(trainY, predictions).eval(session=keras.backend.get_session()).mean()

        return TimeMSEInstance(self)
