import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate = 0.01):
        self.L = len(hidden_layers) # Number of hidden_layers
        self.parameters = {}
        self.learning_rate = learning_rate

        # Initialize weights and biases
        layer_dims = [input_size] + hidden_layers + [output_size]

        for l in range(1, len(layer_dims)):
            self.parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * (2. / layer_dims[l-1])
            self.parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
        return exp_Z / exp_Z.sum(axis = 0, keepdims = True)

    def forward_propagation(self, X):
        cache = {'A0': X} # Input Layer
        A_prev = X # Start with the input
        
        # Loop through hidden_layers (1 to self.L)
        for l in range(1, self.L + 1): # l = 1, 2, 3
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A_prev) + b
            A = self.relu(Z)
            
            # Store values for backprop
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
            A_prev = A # pass to next layer

        # Output Layer (1 = self.L + 1; softmax)
        W = self.parameters[f'W{self.L + 1}']
        b = self.parameters[f'b{self.L + 1}']
        Z = np.dot(W, A_prev) + b
        A = self.softmax(Z)

        cache[f'Z{self.L + 1}'] = Z
        cache[f'A{self.L + 1}'] = A

        return cache

    def compute_loss(self, AL, Y):
        m = Y.shape[1]
        return -np.sum(Y * np.log(AL + 1e-8)) / m

    def backward_propagation(self, cache, Y):
        grads = {}
        m = Y.shape[1]

        # Output Layer gradient
        dZ = cache[f'A{self.L + 1}'] - Y
        grads[f'dW{self.L + 1}'] = np.dot(dZ, cache[f'A{self.L}'].T) / m
        grads[f'db{self.L + 1}'] = np.sum(dZ, axis = 1, keepdims = True) / m
        
        # Hidden Layer gradient
        for l in range(self.L, 0, -1):
            dA_prev = np.dot(self.parameters[f'W{l + 1}'].T, dZ)
            dZ = dA_prev * self.relu_derivative(cache[f'Z{l}'])

            grads[f'dW{l}'] = np.dot(dZ, cache[f'A{l - 1}'].T) / m
            grads[f'db{l}'] = np.sum(dZ, axis = 1, keepdims = True) / m

        return grads

    def update_parameters(self, grads):
        for l in range(1, self.L + 2):
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']

    def train(self, X_train, Y_train, epochs = 1000, print_loss = True):
        for epoch in range(epochs):
            # forward_propagation
            cache = self.forward_propagation(X_train)
            AL = cache[f'A{self.L + 1}']

            # compute_loss
            loss = self.compute_loss(AL, Y_train)

            # backward_propagation
            grads = self.backward_propagation(cache, Y_train)

            # update_parameters
            self.update_parameters(grads)

            if print_loss and epoch % 100 == 0:
                print(f'Epoch { epoch} loss: {loss:.4f}')

    def predict(self, X):
        cache = self.forward_propagation(X)
        AL = cache[f'A{self.L + 1}'] # Output layer activation (softmax probabilities)
        
        predictions = np.argmax(AL, axis = 0)
        return predictions.reshape(1, -1)