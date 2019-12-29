import random

import numpy as np

ACTIVATION_LINEAR = lambda x: x
ACTIVATION_TANH = lambda x: np.tanh(x)
ACTIVATION_SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-x))
ACTIVATION_STEP = lambda x: 1 if x >= 0 else 0

DERIVATIVE_LINEAR = lambda x: 1
DERIVATIVE_TANH = lambda x: 1.0 - x ** 2
DERIVATIVE_SIGMOID = lambda x: ACTIVATION_SIGMOID(x) * (1.0 - ACTIVATION_SIGMOID(x))
DERIVATIVE_STEP = lambda x: 0 if abs(x) < 10 ** -4 else 1

COST_QUADRATIC = lambda x, y: 0.5*(y-x)**2
DERIVATIVE_QUADRATIC = lambda x, y: x-y

reshape_array_to_matrix = lambda a: a.reshape(a.size, 1)


class Network:

    def __init__(self, dimensions, biases=None, weights=None):
        self.activ_func = ACTIVATION_SIGMOID
        self.activ_deriv = DERIVATIVE_SIGMOID
        self.cost_func = COST_QUADRATIC
        self.cost_deriv = DERIVATIVE_QUADRATIC

        self.n_layers = len(dimensions)
        self.dimensions = dimensions
        self.biases = biases
        self.weights = weights

        # biases have a dimension of n_neurons * 1, since we only have 1 bias for each neuron
        if not biases:
            self.biases = [np.random.randn(k) for k in dimensions[1:]]

        # the weights are matrices with dimensions of n_neurons(l) x n_neurons(l+1)
        # we index backwards (j, k) to optimize for matrix transformation
        if not weights:
            self.weights = [np.random.randn(j, k) for k, j in zip(dimensions[:-1], dimensions[1:])]

    def feedforward_la(self, activation):
        for b, w in zip(self.biases, self.weights):
            activation = self.activ_func(np.dot(w, activation) + b)  # a = f(w . a + b)
        return activation

    def feedforward_aa(self, x):
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            activation = self.activ_func(z)  # a = f(w . a + b)
            zs.append(z)
            activations.append(activation)

        return activations, zs

    def train(self, training_data, epochs, batch_size, eta, test_data=None):
        for epoch in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[batch:batch+batch_size] for batch in range(0, len(training_data), batch_size)]

            print(f"Training epoch: {epoch}/{epochs}")
            b = 0
            for batch in batches:
                # print(f"Training batch: {b}")
                self.train_batch(batch, eta)
                b += 1

            if test_data:
                results = self.test(test_data)
                print(f"Accuracy: {results}")

    def train_batch(self, batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        batch_size = len(batch)

        for example, answer in batch:
            # we first calculate the cost gradient for each training example, this is done through backprop
            delta_nabla_b, delta_nabla_w = self.backprop(example, answer)

            # we then add that cost gradient for one training example to the sum of all cost gradients
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # eta/batchsize finds the average gradient
        # we subtract the gradient vector from our weights/biases to take a new learning step
        self.biases = [b-eta/batch_size*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-eta/batch_size*nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward the input, and retrieve activations/weighted sums for each layer
        activations, zs = self.feedforward_aa(x)

        # backprop
        error = self.cost_deriv(activations[-1], y)*self.activ_deriv(zs[-1])

        error_reshaped = reshape_array_to_matrix(error)
        activations_reshaped = reshape_array_to_matrix(activations[-2])

        nabla_b[-1] = error
        nabla_w[-1] = np.dot(error_reshaped, activations_reshaped.transpose())

        for l in range(2, self.n_layers):
            w = self.weights[-l+1]
            z = zs[-l]
            error = np.dot(np.transpose(w), error) * self.activ_deriv(z)
            nabla_b[-l] = error

            error_reshaped = reshape_array_to_matrix(error)
            activations_reshaped = reshape_array_to_matrix(activations[-l-1])

            nabla_w[-l] = np.dot(error_reshaped, activations_reshaped.transpose())

        return nabla_b, nabla_w

    def test(self, test_data):
        success = sum([np.argmax(self.feedforward_la(test[0])) == test[1] for test in test_data])
        return success, len(test_data)

    def test_image(self, test_image):
        return np.argmax(self.feedforward_la(test_image))

    def export_net(self, path):
        import time
        import os

        path = path if not os.path.isdir(path) else f"{path}/network_v{int(time.time())}"
        np.save(path, np.array([self.biases, self.weights, self.dimensions]))

    @staticmethod
    def load_net(path):
        biases, weights, dimensions = np.load(path, allow_pickle=True)
        return Network(dimensions, biases, weights)
