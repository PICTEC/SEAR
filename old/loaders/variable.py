class Variable:
    def __init__(self, key):
        self.key = key

    @property
    def code(self):
        raise NotImplementedError

class FFTVariable(Variable):
    @property
    def code(self):
        return "{} = np.fft.rfft(signal)".format(self.key)
