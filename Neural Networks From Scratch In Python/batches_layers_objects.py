# P.4 Batches, Layers and Objects
import numpy as np

# inputs = [
#     [1, 2, 3, 2.5], 
#     [2., 5., -1., 2], 
#     [-1.5, 2.7, 3.3, -0.8]
# ]
#
# weights = [
#     [0.2, 0.8, -0.5, 1],
#     [0.5, -0.91, 0.26, -0.5],
#     [-0.26, -0.27, 0.17, 0.87]
# ]
# biases = [2, 3, 0.5]
#
# weights2 = [
#     [0.1, -0.14, 0.5],
#     [-0.5, 0.12, -0.33],
#     [-0.44, 0.73, -0.13]
# ]
# biases2 = [-1, 2, -0.5]
#
# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
#
# print(layer2_outputs)

# Input data
X = np.array([
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
])

# LayerDense class definition
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.outputs

    def __str__(self):
        return f"Weights:\n{self.weights}\nBiases:\n{self.biases}"

# Define first layer dimensions
n_inputs = X.shape[1]  # Number of features in input (columns in X)
n_neurons_layer1 = 3   # Desired number of neurons for layer 1

# Define second layer dimensions
n_neurons_layer2 = 5   # Desired number of neurons for layer 2

# Initialize layers
layer1 = LayerDense(n_inputs, n_neurons_layer1)
layer2 = LayerDense(n_neurons_layer1, n_neurons_layer2)

# Perform forward pass through layer 1
output_layer1 = layer1.forward(X)
print("Forward Output Layer 1:")
print(output_layer1)

# Perform forward pass through layer 2
output_layer2 = layer2.forward(output_layer1)
print("\nForward Output Layer 2:")
print(output_layer2)
