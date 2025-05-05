import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, 
                 lambda_reg=0.01, keep_prob=0.8, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                 use_batch_norm=True, use_mini_batch=True):
        # Store the number of hidden layers
        self.L = len(hidden_layers)
        # Dictionary to store weights and biases
        self.parameters = {}
        self.learning_rate = learning_rate
        # L2 regularization parameter
        self.lambda_reg = lambda_reg
        # Dropout keep probability (percentage of neurons to keep during training)
        self.keep_prob = keep_prob
        # Adam hyperparameters
        self.beta1 = beta1         # Decay rate for momentum
        self.beta2 = beta2         # Decay rate for squared gradients
        self.epsilon = epsilon     # Small value to avoid division by zero
        self.is_training = True    # Flag to enable/disable dropout during prediction
        self.t = 0                 # Time step counter for bias correction
        # Batch normalization flag
        self.use_batch_norm = use_batch_norm
        # Mini-batch flag
        self.use_mini_batch = use_mini_batch
        
        # Initialize weights and biases for all layers
        # layer_dims contains sizes of all layers including input and output
        self.layer_dims = [input_size] + hidden_layers + [output_size]
        
        for l in range(1, len(self.layer_dims)):
            # He initialization for better gradient flow in deep networks
            # W shape: (current_layer_size, previous_layer_size)
            self.parameters[f"W{l}"] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2. / self.layer_dims[l-1])
            # Initialize biases to zeros, shape: (current_layer_size, 1)
            self.parameters[f"b{l}"] = np.zeros((self.layer_dims[l], 1))
            
            # Initialize batch normalization parameters for hidden layers
            if self.use_batch_norm and l < len(self.layer_dims) - 1:  # No batch norm for output layer
                # gamma: scale parameter (initially 1)
                self.parameters[f"gamma{l}"] = np.ones((self.layer_dims[l], 1))
                # beta: shift parameter (initially 0)
                self.parameters[f"beta{l}"] = np.zeros((self.layer_dims[l], 1))
                # Running mean and variance for inference (moving averages)
                self.parameters[f"running_mean{l}"] = np.zeros((self.layer_dims[l], 1))
                self.parameters[f"running_var{l}"] = np.zeros((self.layer_dims[l], 1))
        
        # Initialize moment and velocity caches for Adam
        self.m_cache, self.v_cache = self.initialize_adam_caches()

    def initialize_adam_caches(self):
        # Initialize first moment (momentum) and second moment (velocity) for Adam
        m_cache = {}
        v_cache = {}
        for l in range(1, len(self.layer_dims)):
            m_cache[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
            m_cache[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])
            v_cache[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
            v_cache[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])
            
            # Add Adam caches for batch norm parameters
            if self.use_batch_norm and l < len(self.layer_dims) - 1: # No batch norm for output layer
                m_cache[f'dgamma{l}'] = np.zeros_like(self.parameters[f'gamma{l}'])
                m_cache[f'dbeta{l}'] = np.zeros_like(self.parameters[f'beta{l}'])
                v_cache[f'dgamma{l}'] = np.zeros_like(self.parameters[f'gamma{l}'])
                v_cache[f'dbeta{l}'] = np.zeros_like(self.parameters[f'beta{l}'])
                
        return m_cache, v_cache

    def relu(self, Z):
        # ReLU activation function: max(0, Z)
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        # Derivative of ReLU: 1 if Z > 0, else 0
        return np.where(Z > 0, 1, 0)

    def sigmoid(self, Z):
        # Sigmoid activation function for binary classification
        return 1 / (1 + np.exp(-Z))
    
    def sigmoid_derivative(self, Z):
        # Derivative of sigmoid: sigmoid(Z) * (1 - sigmoid(Z))
        s = self.sigmoid(Z)
        return s * (1 - s)
        
    def batch_norm_forward(self, Z, gamma, beta, layer_idx, epsilon=1e-8):
        
        m = Z.shape[1]  # Number of examples in batch
        
        # Compute mean and variance
        mu = np.mean(Z, axis=1, keepdims=True)      # (n_features, 1)
        var = np.var(Z, axis=1, keepdims=True)      # (n_features, 1)
        
        # If in training mode, update running statistics with momentum (0.9)
        if self.is_training:
            momentum = 0.9
            self.parameters[f"running_mean{layer_idx}"] = (
                momentum * self.parameters[f"running_mean{layer_idx}"] + 
                (1 - momentum) * mu
            )
            self.parameters[f"running_var{layer_idx}"] = (
                momentum * self.parameters[f"running_var{layer_idx}"] + 
                (1 - momentum) * var
            )
        
        # Use running statistics during inference
        if not self.is_training:
            mu = self.parameters[f"running_mean{layer_idx}"]
            var = self.parameters[f"running_var{layer_idx}"]
        
        # Normalize
        Z_norm = (Z - mu) / np.sqrt(var + epsilon)
        
        # Scale and shift
        Z_tilde = gamma * Z_norm + beta
        
        # Cache values for backward pass
        cache = {
            "Z": Z,
            "mu": mu,
            "var": var,
            "Z_norm": Z_norm,
            "gamma": gamma,
            "beta": beta,
            "epsilon": epsilon
        }
        
        return Z_tilde, cache

    def batch_norm_backward(self, dZ_tilde, cache):

        Z = cache["Z"]
        mu = cache["mu"]
        var = cache["var"]
        Z_norm = cache["Z_norm"]
        gamma = cache["gamma"]
        epsilon = cache["epsilon"]
        m = Z.shape[1]
        
        # Compute gradients
        dgamma = np.sum(dZ_tilde * Z_norm, axis=1, keepdims=True)
        dbeta = np.sum(dZ_tilde, axis=1, keepdims=True)
        
        # Compute gradient with respect to Z_norm
        dZ_norm = dZ_tilde * gamma
        
        # Compute gradient with respect to Z using the batch norm equations
        dvar = np.sum(dZ_norm * (Z - mu) * -0.5 * np.power(var + epsilon, -1.5), axis=1, keepdims=True)
        dmu = np.sum(dZ_norm * -1 / np.sqrt(var + epsilon), axis=1, keepdims=True) + dvar * np.sum(-2 * (Z - mu), axis=1, keepdims=True) / m
        dZ = dZ_norm / np.sqrt(var + epsilon) + dvar * 2 * (Z - mu) / m + dmu / m
        
        return dZ, dgamma, dbeta

    def forward_propagation(self, X):
        # Cache stores all activations, Z values, and batch norm caches for backpropagation
        cache = {'A0': X}  # Input layer activation
        A_prev = X  # Current activation to pass forward
        
        # Loop through hidden layers (1 to self.L)
        for l in range(1, self.L + 1): 
            W = self.parameters[f'W{l}']  # Weights of current layer
            b = self.parameters[f'b{l}']  # Biases of current layer
            
            # Linear transformation: Z = W·A + b
            Z = np.dot(W, A_prev) + b
            
            # Apply batch normalization if enabled (for hidden layers)
            if self.use_batch_norm and l < self.L + 1:  # No batch norm at output layer
                gamma = self.parameters[f'gamma{l}']
                beta = self.parameters[f'beta{l}']
                Z_tilde, bn_cache = self.batch_norm_forward(Z, gamma, beta, l)
                cache[f'bn_cache{l}'] = bn_cache
                # Use normalized Z for activation
                Z = Z_tilde
            
            # Store pre-activation values
            cache[f'Z{l}'] = Z
            
            # Apply ReLU activation to hidden layers
            A = self.relu(Z)
            
            # Apply dropout only during training
            if self.is_training and l < self.L + 1:  # Apply dropout to all layers except output
                # Create dropout mask (1: keep, 0: drop)
                cache[f'D{l}'] = np.random.rand(*A.shape) < self.keep_prob
                # Apply dropout mask and scale by keep_prob to maintain expected values
                A = A * cache[f'D{l}'] / self.keep_prob
            
            # Store post-activation
            cache[f'A{l}'] = A
            A_prev = A  # This layer's output becomes next layer's input

        # Output Layer (layer L+1) uses sigmoid activation for binary classification
        W = self.parameters[f'W{self.L + 1}']
        b = self.parameters[f'b{self.L + 1}']
        Z = np.dot(W, A_prev) + b
        A = self.sigmoid(Z)  # Use sigmoid for binary classification

        cache[f'Z{self.L + 1}'] = Z
        cache[f'A{self.L + 1}'] = A

        return cache

    def compute_loss(self, AL, Y):
        # Binary cross-entropy loss for sigmoid output
        m = Y.shape[1]  # Number of examples
        
        # Compute binary cross-entropy loss
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        loss = -1/m * np.sum(Y * np.log(AL + epsilon) + (1 - Y) * np.log(1 - AL + epsilon))
        
        # L2 regularization term (sum of squared weights)
        l2_reg_cost = 0
        for l in range(1, self.L + 2):  # Include all layers
            l2_reg_cost += np.sum(np.square(self.parameters[f'W{l}']))
        
        # Add L2 regularization term to the loss
        l2_reg_cost = (self.lambda_reg / (2 * m)) * l2_reg_cost
        
        # Total cost = cross-entropy loss + L2 regularization
        return loss + l2_reg_cost

    def backward_propagation(self, cache, Y):
        grads = {}  # Store gradients for all parameters
        m = Y.shape[1]  # Number of examples

        # Output Layer gradient calculation for sigmoid with binary cross-entropy
        AL = cache[f'A{self.L + 1}']
        dZ = AL - Y  # For sigmoid with cross-entropy
        
        # Gradient of weights: dW = (dZ · previous_activation_transpose) / m + lambda * W / m (L2 term)
        grads[f'dW{self.L + 1}'] = (np.dot(dZ, cache[f'A{self.L}'].T) / m) + ((self.lambda_reg / m) * self.parameters[f'W{self.L + 1}'])
        # Gradient of biases: sum dZ across examples
        grads[f'db{self.L + 1}'] = np.sum(dZ, axis=1, keepdims=True) / m
        
        # Hidden Layers gradient calculation (backpropagating the error)
        for l in range(self.L, 0, -1):  # Loop backwards through hidden layers
            # Propagate error from layer l+1 to layer l
            dA_prev = np.dot(self.parameters[f'W{l + 1}'].T, dZ)
            
            # If dropout was applied during forward pass, apply the same mask
            if self.is_training:
                dA_prev = dA_prev * cache[f'D{l}'] / self.keep_prob
                
            # Apply ReLU derivative to get dZ for current layer
            dZ = dA_prev * self.relu_derivative(cache[f'Z{l}'])
            
            # Apply batch normalization backward pass if enabled
            if self.use_batch_norm and l < self.L + 1:     # No batch norm for output layer    
                dZ, dgamma, dbeta = self.batch_norm_backward(dZ, cache[f'bn_cache{l}'])
                # Store gradients for batch norm parameters
                grads[f'dgamma{l}'] = dgamma
                grads[f'dbeta{l}'] = dbeta
            
            # Calculate gradients for weights with L2 regularization
            grads[f'dW{l}'] = (np.dot(dZ, cache[f'A{l - 1}'].T) / m) + ((self.lambda_reg / m) * self.parameters[f'W{l}'])
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m

        return grads

    def update_parameters(self, grads):
        # Update all weights and biases using Adam optimization
        # Increment time step for bias correction
        self.t += 1     
        
        for l in range(1, self.L + 2):  # +2 because we include output layer
            # Update weights and biases
            self._update_parameter_with_adam(f'W{l}', grads[f'dW{l}'])
            self._update_parameter_with_adam(f'b{l}', grads[f'db{l}'])
            
            # Update batch norm parameters if applicable
            if self.use_batch_norm and l < self.L + 1:  # No batch norm for output layer
                self._update_parameter_with_adam(f'gamma{l}', grads[f'dgamma{l}'])
                self._update_parameter_with_adam(f'beta{l}', grads[f'dbeta{l}'])
    
    def _update_parameter_with_adam(self, param_name, grad):
        """Helper method to update a parameter using Adam optimization"""
        # Update first moment (momentum)
        self.m_cache[f'd{param_name}'] = self.beta1 * self.m_cache[f'd{param_name}'] + (1 - self.beta1) * grad
        
        # Update second moment (velocity)
        self.v_cache[f'd{param_name}'] = self.beta2 * self.v_cache[f'd{param_name}'] + (1 - self.beta2) * np.square(grad)
        
        # Bias correction
        m_hat = self.m_cache[f'd{param_name}'] / (1 - self.beta1**self.t)
        v_hat = self.v_cache[f'd{param_name}'] / (1 - self.beta2**self.t)
        
        # Update parameter
        self.parameters[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def create_mini_batches(self, X, Y, batch_size):
        m = X.shape[1]
        mini_batches = []
        
        # Shuffle data
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        # Create mini-batches
        num_batches = m // batch_size
        for i in range(num_batches):
            X_batch = X_shuffled[:, i * batch_size: (i + 1) * batch_size]
            Y_batch = Y_shuffled[:, i * batch_size: (i + 1) * batch_size]
            mini_batches.append((X_batch, Y_batch))
        
        # Handle remaining samples if they don't fit perfectly into batches
        if m % batch_size != 0:
            X_batch = X_shuffled[:, num_batches * batch_size:]
            Y_batch = Y_shuffled[:, num_batches * batch_size:]
            mini_batches.append((X_batch, Y_batch))

        return mini_batches

    def train(self, X_train, Y_train, epochs=1000, batch_size=32, print_loss=True, save_loss_graph=True):
        self.is_training = True  # Enable dropout during training
        
        # Store loss history for plotting
        loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            if self.use_mini_batch:
                # Create mini-batches
                mini_batches = self.create_mini_batches(X_train, Y_train, batch_size)
                
                # Process each mini-batch
                for mini_batch in mini_batches:
                    X_batch, Y_batch = mini_batch
                    
                    # Step 1: Forward propagation - compute activations
                    cache = self.forward_propagation(X_batch)
                    AL = cache[f'A{self.L + 1}']  # Output layer activation

                    # Step 2: Compute loss
                    batch_loss = self.compute_loss(AL, Y_batch)
                    epoch_loss += batch_loss * (X_batch.shape[1] / X_train.shape[1])  # Weighted by batch size

                    # Step 3: Backward propagation - compute gradients
                    grads = self.backward_propagation(cache, Y_batch)

                    # Step 4: Update parameters using Adam
                    self.update_parameters(grads)
            else:
                # Step 1: Forward propagation - compute activations
                cache = self.forward_propagation(X_train)
                AL = cache[f'A{self.L + 1}']  # Output layer activation

                # Step 2: Compute loss
                epoch_loss = self.compute_loss(AL, Y_train)

                # Step 3: Backward propagation - compute gradients
                grads = self.backward_propagation(cache, Y_train)

                # Step 4: Update parameters using Adam
                self.update_parameters(grads)

            # Store average loss for this epoch
            loss_history.append(epoch_loss)
            
            # Print loss every 100 epochs if requested
            if print_loss and epoch % 100 == 0:
                print(f'Epoch {epoch} loss: {epoch_loss:.4f}')
                
        # Plot the loss curve
        if print_loss or save_loss_graph:
            plt.figure(figsize=(10, 6))
            plt.plot(range(epochs), loss_history)
            plt.title(f'Loss vs. Epochs (Batch Size: {batch_size})')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            
            if save_loss_graph:
                # Create directory if it doesn't exist
                import os
                os.makedirs('plots', exist_ok=True)
                plt.savefig(f'plots/loss_curve_batch_{batch_size}.png')
                print(f"Loss graph saved to plots/loss_curve_batch_{batch_size}.png")
            
            if print_loss:
                plt.show()
        
        return loss_history

    def predict(self, X):
        # Disable dropout during prediction
        self.is_training = False
        
        # Forward pass to get output probabilities
        cache = self.forward_propagation(X)
        AL = cache[f'A{self.L + 1}']  # Output layer activation (sigmoid probabilities)
        
        # Get the class predictions (1 if probability > 0.5, 0 otherwise)
        predictions = (AL > 0.5).astype(int)
        return predictions

# Calculate accuracy
def calculate_accuracy(predictions, y):
    return np.mean(predictions == y) * 100

# Function to train and evaluate the model on cat vs non-cat dataset
def train_cat_classifier(train_x, train_y, test_x, test_y, classes, print_images=True):
    print("Training Neural Network for Cat vs Non-Cat Classification")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(1)
    
    # Get data dimensions
    n_x = train_x.shape[0]  # Input size (number of features)
    n_y = 1                 # Output size (binary classification)
    
    # Define network architecture
    hidden_layers = [20, 7, 5]  # Hidden layer sizes
    learning_rate = 0.0075
    lambda_reg = 0.01    # L2 regularization parameter
    keep_prob = 0.8      # Dropout keep probability
    use_batch_norm = True
    use_mini_batch = True
    
    # Create neural network
    nn = NeuralNetwork(n_x, hidden_layers, n_y, learning_rate, lambda_reg, 
                      keep_prob, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                      use_batch_norm=use_batch_norm, use_mini_batch=use_mini_batch)
    
    # Train the network
    print("\nTraining neural network...")
    nn.train(train_x, train_y, epochs=2000, batch_size=32, print_loss=True)
    
    # Make predictions
    train_predictions = nn.predict(train_x)
    test_predictions = nn.predict(test_x)
    
    # Calculate accuracies
    train_accuracy = calculate_accuracy(train_predictions, train_y)
    test_accuracy = calculate_accuracy(test_predictions, test_y)
    
    print(f"\nTraining accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Print examples of predictions if requested
    if print_images:
        print("\nViewing some predictions:")
        num_samples_to_show = min(5, test_x.shape[1])
        
        for i in range(num_samples_to_show):
            plt.figure(figsize=(2, 2))
            # Reshape the flattened image back to (64, 64, 3)
            img = test_x[:, i].reshape((64, 64, 3))
            plt.imshow(img)
            plt.title(f"Prediction: {classes[int(test_predictions[0, i])].decode('utf-8')}")
            plt.axis('off')
            plt.show()
            print(f"True label: {classes[int(test_y[0, i])].decode('utf-8')}")
            print("-" * 30)
    
    return nn, train_accuracy, test_accuracy