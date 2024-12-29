import numpy as np

class NN:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters()
        self.L = len(layer_dims) - 1  # number of layers excluding input
        print(f"Number of layers: {self.L}")

    def initialize_parameters(self):
        np.random.seed
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2. / self.layer_dims[l-1])
            parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
            print(f"Layer {l}: W{l} shape: {parameters[f'W{l}'].shape}, b{l} shape: {parameters[f'b{l}'].shape}")
        
        print(f"Parameters initialized: {parameters.keys()}")
        print(f"Parameters; {parameters}")
        return parameters

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    # def sigmoid(self, Z):
    #     return 1 / 1 + np.exp(-Z)

    @staticmethod
    def relu_backward(dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    @staticmethod
    def softmax(Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    @staticmethod
    def softmax_backward(dA, cache):
        Z = cache
        s = NN.softmax(Z)
        dZ = dA * s * (1 - s)
        return dZ

    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        print(f"A_prev shape: {A_prev.shape}, W shape: {W.shape}, b shape: {b.shape}")
        print(f"A_prev: {A_prev}, W: {W}, b: {b}")
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        
        if activation == "relu":
            A, activation_cache = self.relu(Z), Z
        elif activation == "softmax":
            A, activation_cache = self.softmax(Z), Z
        
        cache = (linear_cache, activation_cache)
        return A, cache

    def forward_propagation(self, X):
        caches = []
        A = X
        
        print(f"cahces: {caches}")
        print(f"A: {A}")
        
        # Hidden layers
        for l in range(1, self.L):
            A_prev = A
            print(f"Layer {l}")
            print(f"A_prev: {A_prev}")
            A, cache = self.linear_activation_forward(A_prev, self.parameters[f'W{l}'], self.parameters[f'b{l}'], activation="relu")
            print(f"A: {A}")
            print(f"Cache: {cache}")
            caches.append(cache)
            
        # Output layer
        AL, cache = self.linear_activation_forward(A, self.parameters[f'W{self.L}'], self.parameters[f'b{self.L}'], activation="softmax")
        print(f"Layer {self.L}")
        print(f"AL: {AL}")
        print(f"Cache: {cache}")
        caches.append(cache)
        
        return AL, caches

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        print(f"Activation: {activation}")
        print(f"dA: {dA}")
        print(f"Cache: {cache}")
        linear_cache, activation_cache = cache
        print(f"Linear cache: {linear_cache}")
        print(f"Activation cache: {activation_cache}")
        
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            
            print(f"dZ: {dZ}")
            print(f"dA_prev: {dA_prev}")
            print(f"dW: {dW}")
            print(f"db: {db}")
            
        elif activation == "softmax":
            dZ = self.softmax_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            print(f"dZ: {dZ}")
            print(f"dA_prev: {dA_prev}")
            print(f"dW: {dW}")
            print(f"db: {db}")
        
        print(f"dA_prev: {dA_prev}")
        print(f"dW: {dW}")
        print(f"db: {db}")
        
        return dA_prev, dW, db

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        m = AL.shape[1]  # Number of examples
        Y = Y.reshape(AL.shape)  # Ensure Y has the same shape as AL
        
        # Initialize the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        current_cache = caches[self.L - 1]
        print(f"Current cache: {current_cache}")
        grads[f"dA{self.L-1}"], grads[f"dW{self.L}"], grads[f"db{self.L}"] = self.linear_activation_backward(dAL, current_cache, activation="softmax")
        
        for l in reversed(range(self.L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads[f"dA{l+1}"], current_cache, activation="relu")
            grads[f"dA{l}"] = dA_prev_temp
            grads[f"dW{l+1}"] = dW_temp
            grads[f"db{l+1}"] = db_temp

            print(f"Layer {l}")
            print(f"dA_prev_temp: {dA_prev_temp}")
            print(f"dW_temp: {dW_temp}")
            print(f"db_temp: {db_temp}")
        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']

    def train(self, X, Y, num_iterations, learning_rate):
        costs = []
        for i in range(num_iterations):
            # Forward propagation
            AL, caches = self.forward_propagation(X)
            
            # Compute cost
            cost = -np.sum(Y * np.log(AL + 1e-8)) / Y.shape[1]
            
            # Backward propagation
            grads = self.backward_propagation(AL, Y, caches)
            
            # Update parameters
            self.update_parameters(grads, learning_rate)
            
            if i % 100 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost}")
        
        return costs

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        return np.argmax(AL, axis=0)
