
import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
import nnfs

# Initialize nnfs to ensure consistent results
nnfs.init()

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
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        proba = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = proba

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class CategoricalCrossEntropyLoss(Loss):
    def forward(self, y_prediction, y_true):
        samples = len(y_prediction)
        y_pred_clip = np.clip(y_prediction, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clip[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clip * y_true, axis=1)

        negative_log = -np.log(correct_confidence)
        return negative_log

def accuracy(softmax_outputs, targets):
    predictions = np.argmax(softmax_outputs, axis=1)
    return np.mean(predictions == targets)


# Create a dense layer with 2 inputs (features) and 3 neurons
dense1 = LayerDense(2, 3)
activation1 = ActivationReLU()

dense2 = LayerDense(3, 3)
activation2 = ActivationSoftMax()

# Forward pass
dense1.forward(X)
activation1.forward(dense1.outputs)

dense2.forward(activation1.outputs)
activation2.forward(dense2.outputs)

# Calculate loss
loss_function = CategoricalCrossEntropyLoss()
loss = loss_function.calculate(activation2.outputs, y)

# Print loss
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy(activation2.outputs, y)}")
