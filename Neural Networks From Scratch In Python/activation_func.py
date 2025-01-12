
import numpy as np
from nnfs.datasets import spiral_data
# Input data
X, y = spiral_data(100, 3)

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def __str__(self):
        return f"Weights:\n{self.weights}\nBiases:\n{self.biases}"

class ActivationReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

# Create a dense layer with 2 inputs (features) and 5 neurons
layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()

# Forward pass through the dense layer
layer1.forward(X)
print("Layer 1 outputs (before activation):")
print(layer1.outputs)

# Forward pass through ReLU activation
activation1.forward(layer1.outputs)
print("Layer 1 outputs (after activation):")
print(activation1.outputs)

