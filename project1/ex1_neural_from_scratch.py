import learn_nump as np

inputs = [1, 2, 3, 2.5]
# 3 layers
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# layer_output = weight * input + bias
output = np.dot(weights, inputs) + biases
print(output)

# output2 = np.dot(inputs, weights) + biases
# print(output2)
# ValueError: shapes (4,) and (3,4) not aligned: 4 (dim 0) != 3 (dim 0)

inputs2 = [[1, 2, 3, 2.5],
           [2.0, 5.0, -1.0, 2.0],
           [-1.5, 2.7, 3.3, -0.8]]

outputs = np.dot(inputs2, np.array(weights).T) + biases
print(f"Output2: \n{outputs}")

weights3 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases3 = [-1, 2, -0.5]

outputs3 = np.dot(np.array(inputs2).T, weights3) + biases3
print(f"Output3: \n{outputs3}")

layer1_outputs = np.dot(inputs2, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights3).T) + biases3
print(f"layer 2 outputs: \n{layer2_outputs}")