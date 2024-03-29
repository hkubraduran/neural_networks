import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.output = probabilities

X,y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

dense2 = LayerDense(3, 3)
activation2 = ActivationSoftmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])