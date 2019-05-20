import keras
import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D, Lambda


class CLR(keras.optimizers.Optimizer):
    """
    Supports clipnorm clipvalue
     bounds - a pair of lower and upper learning_rate
     step_function - 'triangular', 'sinusoidal_pulse'
    """

    step_functions = {
        'triangular': lambda i, bounds, steps: K.control_flow_ops.cond(K.equal((i // steps) % 2, 0),
            lambda: (bounds[1] - ((i % steps)) * ((bounds[1] - bounds[0]) // (steps - 1))),
            lambda: (bounds[0] + ((i % steps)) * ((bounds[1] - bounds[0]) // (steps - 1))))
        }

    def __init__(self, bounds=(0.01, 3.), steps=15, momentum_bounds=0., step_function='triangular', **kwargs):
        super(CLR, self).__init__(**kwargs)
        self.learning_rate_bounds = K.constant(bounds)
        self.momentum_bounds = K.constant(momentum_bounds if type(momentum_bounds) in [tuple, list] else (momentum_bounds, momentum_bounds))
        self.steps = K.constant(steps)
        self.step_function = self.step_functions[step_function]
        self.lr = K.variable(bounds[0], name='lr')
        self.momentum = K.variable(momentum_bounds[0] if type(momentum_bounds) in [tuple, list] else momentum_bounds, name='momentum')
        self.iterations = K.variable(0, name='iterations')
        K.get_session().run(self.iterations.initializer)

    def get_updates(self, params, constraints, loss):
        gradients = self.get_gradients(loss, params)
        self.updates = []
        self._update_runtime_parameters()  # this needs to be integrated into the loop...
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        for param, grad, moment in zip(params, gradients, moments):
            velocity = self.momentum * moment - self.lr * grad
            self.updates.append(K.update(moment, velocity))
            new_param = param + velocity
            if param in constraints.keys():
                constraint = constraints[param]
                new_param = constraint(new_param)
            self.updates.append(K.update(param, new_param))
        return self.updates

    def get_config(self):
        config = {'learning_rate_bounds': self.learning_rate_bounds,
                    'momentum_bounds': self.momentum_bounds,
                    'steps': self.steps,
                    'step_function': self.step_function}
        base_config = super(CLR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _update_runtime_parameters(self):
        self.lr = K.update(self.lr, K.variable(self.step_function(self.iterations, self.learning_rate_bounds, self.steps)))
        self.momentum = K.update(self.momentum, K.variable(self.step_function(self.iterations, self.momentum_bounds, self.steps)))
        self.updates.append(K.update(self.iterations, self.iterations + 1))

def MNIST_model():
    input_layer = Input([28, 28])
    layer0 = Lambda(lambda x:K.expand_dims(x,3))(input_layer)
    layer1 = Conv2D(32, kernel_size=3, strides = 2, activation = 'relu')(layer0)
    layer3 = Conv2D(16, kernel_size=3, activation = 'relu')(layer1)
    layer5 = Conv2D(20, kernel_size=3, strides = 2, activation = 'relu')(layer3)
    flatten = Flatten()(layer5)
    dense1 = Dense(64)(flatten)
    dense2 = Dense(10, activation = 'softmax')(dense1)
    m = keras.models.Model(input_layer, dense2)
    return m

import keras
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense, Input, concatenate
from keras.datasets import cifar10

def CIFAR_model():
	input_layer = Input([32, 32, 3])
	layer0 = Conv2D(128, kernel_size=3, strides=2, padding='same')(input_layer)
	layer1 = BatchNormalization()(layer0)
	layer2 = Activation('relu')(layer1)
	for i in range(8):
		layer3 = Conv2D(32, kernel_size=3, padding='same')(layer2)
		layer4 = BatchNormalization()(layer3)
		layer5 = Activation('relu')(layer4)
		layer6 = Conv2D(128, kernel_size=3, padding='same')(layer5)
		layer7 = BatchNormalization()(layer6)
		layer8 = Activation('relu')(layer7)
		layer9 = Conv2D(128, kernel_size=1)(concatenate([layer8, layer2]))
		layer10 = BatchNormalization()(layer9)
		layer2 = Activation('relu')(layer10)
	conv = Conv2D(128, kernel_size=3)(layer2)
	bn = BatchNormalization()(conv)
	act = Activation('relu')(bn)
	avg = AveragePooling2D(pool_size=14)(act)
	flatten = Flatten()(avg)
	output_layer = Dense(10, activation='softmax')(flatten)
	return Model(input_layer, output_layer)


if __name__=='__main__':

    """
    # Experiment one
    (trainX, trainY), (testX, testY) = mnist.load_data()

    m = MNIST_model()
    m.compile(CLR([0.0005, 0.002], momentum_bounds=[0.95, 0.8], steps= 40), 'categorical_crossentropy')
    m.fit(trainX, keras.utils.to_categorical(trainY), epochs = 3)

    m = MNIST_model()
    m.compile(keras.optimizers.Adam(), 'categorical_crossentropy')
    m.fit(trainX, keras.utils.to_categorical(trainY), epochs = 3)

    m = MNIST_model()
    m.compile(keras.optimizers.SGD(0.001, momentum=0.8), 'categorical_crossentropy')
    m.fit(trainX, keras.utils.to_categorical(trainY), epochs = 3)
    """

    (trainX, trainY), (testX, testY) = cifar10.load_data()

    """
    m = CIFAR_model()
    m.compile(keras.optimizers.Adam(), 'categorical_crossentropy')
    m.fit(trainX, keras.utils.to_categorical(trainY), epochs = 10)
    # has 1.4312 in one epoch, 0.2703 in ten epochs, takes 15min per epoch
    """

    m = CIFAR_model()
    m.compile(CLR([0.1, 3.5], steps=5000), 'categorical_crossentropy')
    m.fit(trainX, keras.utils.to_categorical(trainY), epochs = 8)
    # has 1.6171 in one epoch and 0.3574 in ten

# za czesto FB-  cos z tym zrobic
# Iron supplementation improved verbal learning and supplementation
