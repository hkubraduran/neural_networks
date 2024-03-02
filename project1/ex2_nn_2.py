
import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:

     def __init__(self, n_inputs, n_neurons):
          '''
          randn generates numbers from the standart normal distribution
          The standard normal distribution is a normal distribution that has a mean of 0
          and a standard deviation of 1.
          '''
          self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
          self.biases = np.zeros((1, n_neurons))
     def forward(self, inputs):
          self.output = np.dot(inputs, self.weights) + self.biases
# n_neurons is totally up to you.

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(f"Layer 1 output: \n{layer1.output}")

layer2.forward(layer1.output)
print(f"Layer 2 output: \n{layer2.output}")

# print(np.random.randn(4, 3))
# Some of the elements are bigger than 1
# print(0.10 * np.random.randn(4, 3))

# basic Rectified Linear Unit Activation Function
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
for i in inputs:
     # if i <= 0:
     #     output.append(0)
     # elif i > 0:
     #      output.append(i)
     output.append(max(0, i))
print(output)