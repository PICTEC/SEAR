class Schedule(object):

    default_batch_size = 32

    def __init__(self, rates, epochs, optimizer = None):
        self.optimizer = optimizer
        self.rates = rates
        self.epochs = epochs
        self.batch_size = self.default_batch_size

    def add_model(self, model):
        self.model = model
        if self.optimizer is None:
            self.optimizer = self.model.optimizer.__class__
        return self

    def _fit(self, fit_callback, run_parameters, ignore_batch_size = False):
        hist = []
        if not ignore_batch_size:
            if "batch_size" not in run_parameters:
                run_parameters["batch_size"] = self.batch_size
            else:
                self.batch_size = run_parameters["batch_size"]
        if "epochs" in run_parameters.keys():
            run_parameters.pop("epochs")
        for rate, epochs in zip(self.rates, self.epochs):
            self.model.compile(self.optimizer(rate), self.model.loss, metrics=self.model.metrics)
            hist.append(fit_callback(epochs, run_parameters).history)
        history = {}
        for key in hist[0].keys():
            history[key] = [x for stage in hist for x in stage[key]]
        class QuasiHistory(object):
            def __init__(self, history):
                self.history = history
        return QuasiHistory(history)

    def fit(self, trainX, trainY, validation_data=None, **run_parameters):
        cb = lambda epochs, run_parameters: self.model.fit(trainX, trainY, validation_data=validation_data, epochs = epochs, **run_parameters)
        return self._fit(cb, run_parameters)

    def fit_generator(self, train, steps_per_epoch, validation_data=None, validation_steps = None, **run_parameters):
        def cb(epochs, run_parameters): 
            if "batch_size" in run_parameters.keys(): run_parameters.pop("batch_size")    
            print(run_parameters, validation_steps)
            return self.model.fit_generator(train, int(steps_per_epoch), validation_data=validation_data, validation_steps = int(validation_steps), epochs = int(epochs), **run_parameters)
        return self._fit(cb, run_parameters, ignore_batch_size = True)