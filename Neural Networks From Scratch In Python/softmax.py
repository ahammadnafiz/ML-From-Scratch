
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

class ActivationSoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True))
        proba = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = proba

# Create a dense layer with 2 inputs (features) and 5 neurons
dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

dense2 = LayerDense(3, 3)
activation2 = ActivationSoftMax()

dense1.forward(X)
activation1.forward(dense1.outputs)

dense2.forward(activation1.outputs)
activation2.forward(dense2.outputs)

print(activation2.outputs[:5])


# Visualizing data at different stages
plt.figure(figsize=(12, 8))

# Input data visualization
plt.subplot(2, 2, 1)
plt.title("Input Data")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Output of first dense layer
plt.subplot(2, 2, 2)
plt.title("Dense Layer 1 Outputs")
plt.scatter(dense1.outputs[:, 0], dense1.outputs[:, 1], c=y, cmap='brg')
plt.xlabel("Neuron 1 Output")
plt.ylabel("Neuron 2 Output")

# Output of ReLU activation
plt.subplot(2, 2, 3)
plt.title("ReLU Activation Outputs")
plt.scatter(activation1.outputs[:, 0], activation1.outputs[:, 1], c=y, cmap='brg')
plt.xlabel("Neuron 1 Output (ReLU)")
plt.ylabel("Neuron 2 Output (ReLU)")

# Output of SoftMax activation (probabilities)
plt.subplot(2, 2, 4)
plt.title("SoftMax Activation Outputs")
plt.scatter(activation2.outputs[:, 0], activation2.outputs[:, 1], c=np.argmax(activation2.outputs, axis=1), cmap='brg')
plt.xlabel("Class 1 Probability")
plt.ylabel("Class 2 Probability")

plt.tight_layout()
plt.show()
