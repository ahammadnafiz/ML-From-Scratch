
```python
class NeuralNetwork:
    def __init__(self, layers_dims, learning_rate=0.01):
        self.layers_dims = layers_dims  # List containing layer dimensions
        self.learning_rate = learning_rate
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        parameters = {}
        L = len(self.layers_dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.layers_dims[l], 1))
        return parameters

    def forward_prop(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2
        
        for l in range(1, L):
            A_prev = A 
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A = np.maximum(0, Z)  # ReLU activation
            caches.append((A_prev, W, b, Z))
        
        W = self.parameters['W' + str(L)]
        b = self.parameters['b' + str(L)]
        ZL = np.dot(W, A) + b
        AL = self.softmax(ZL)  # Softmax activation
        caches.append((A, W, b, ZL))
        
        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL)) / m
        return np.squeeze(cost)

    def backward_prop(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = AL - Y
        
        # Last layer gradient (Softmax)
        current_cache = caches[L-1]
        A_prev, W, b, ZL = current_cache
        grads["dW" + str(L)] = np.dot(dAL, A_prev.T) / m
        grads["db" + str(L)] = np.sum(dAL, axis=1, keepdims=True) / m
        grads["dA" + str(L-1)] = np.dot(W.T, dAL)
        
        # Hidden layers gradients (ReLU)
        for l in reversed(range(1, L)):
            current_cache = caches[l-1]
            A_prev, W, b, Z = current_cache
            dZ = np.array(grads["dA" + str(l)]).copy()
            dZ[Z <= 0] = 0  # ReLU derivative
            
            grads["dW" + str(l)] = np.dot(dZ, A_prev.T) / m
            grads["db" + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m
            grads["dA" + str(l-1)] = np.dot(W.T, dZ)
        
        return grads

    def update_parameters(self, grads):
        L = len(self.parameters) // 2
        for l in range(1, L+1):
            self.parameters["W" + str(l)] -= self.learning_rate * grads["dW" + str(l)]
            self.parameters["b" + str(l)] -= self.learning_rate * grads["db" + str(l)]

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

# Simulating training loop for batches of data
def train_neural_network(X_batches, Y_batches, layers_dims, epochs=100):
    nn = NeuralNetwork(layers_dims)

    for epoch in range(epochs):
        total_cost = 0
        for t in range(len(X_batches)):  # Loop over batches (one epoch)
            AL, caches = nn.forward_prop(X_batches[t])
            cost = nn.compute_cost(AL, Y_batches[t])
            grads = nn.backward_prop(AL, Y_batches[t], caches)
            nn.update_parameters(grads)
            total_cost += cost
        
        print(f"Epoch {epoch + 1}/{epochs}, Cost: {total_cost / len(X_batches)}")

    return nn

```