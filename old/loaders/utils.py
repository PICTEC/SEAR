class ProgressBar(object):
    def __init__(self):
        self.max = 100
        self.curr = 0

    def set_max(self, val):
        self.max = val
        self.curr = 0

    def step(self, payload):
        self.curr += 1
        progress = "{}/{}".format(self.curr, self.max)
        print("\x1b[1A\x1b[2K{:20} - {}".format(progress, payload))
