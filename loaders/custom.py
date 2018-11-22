import keras.backend as K
import numpy as np
import tensorflow as tf

class Constr:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.mask = np.repeat(np.eye(X), Y).reshape(X, X, Y).swapaxes(2,0)
        self.mask = K.constant(self.mask)

    def __call__(self, tensor):
        return tensor * self.mask

    def get_config(self):
        return {"X": self.X, "Y": self.Y}

def maximal_error(x, y):
    return K.max(K.abs(y - x))

def identity_loss(true, pred):
    return pred

def l4_loss(true, pred):
    return K.mean(K.square(K.square(pred - true)), axis=-1)

custom_objects = {"Constr": Constr,
                  "maximal_error": maximal_error,
                  "tf": tf,
                  "identity_loss": identity_loss,
                  "l4_loss": l4_loss}
