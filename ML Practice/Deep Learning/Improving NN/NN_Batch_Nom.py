import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, 
                 lambda_reg=0.01, keep_prob=0.8, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                 use_batch_norm=True):
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

    def softmax(self, Z):
        # Softmax activation for output layer (multi-class classification)
        # Subtracting max(Z) for numerical stability to prevent overflow
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        # Normalize to get probability distribution
        return exp_Z / exp_Z.sum(axis=0, keepdims=True)
        
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

        # Output Layer (layer L+1) uses softmax activation
        W = self.parameters[f'W{self.L + 1}']
        b = self.parameters[f'b{self.L + 1}']
        Z = np.dot(W, A_prev) + b
        A = self.softmax(Z)  # Convert to probability distribution

        cache[f'Z{self.L + 1}'] = Z
        cache[f'A{self.L + 1}'] = A

        return cache

    def compute_loss(self, AL, Y):
        # Cross-entropy loss for softmax output
        m = Y.shape[1]  # Number of examples
        # Small epsilon (1e-8) added to prevent log(0)
        cross_entropy_loss = -np.sum(Y * np.log(AL + 1e-8)) / m
        
        # L2 regularization term (sum of squared weights)
        l2_reg_cost = 0
        for l in range(1, self.L + 2):  # Include all layers
            l2_reg_cost += np.sum(np.square(self.parameters[f'W{l}']))
        
        # Add L2 regularization term to the loss
        l2_reg_cost = (self.lambda_reg / (2 * m)) * l2_reg_cost
        
        # Total cost = cross-entropy loss + L2 regularization
        return cross_entropy_loss + l2_reg_cost

    def backward_propagation(self, cache, Y):
        grads = {}  # Store gradients for all parameters
        m = Y.shape[1]  # Number of examples

        # Output Layer gradient calculation
        # For softmax with cross-entropy, gradient is (prediction - actual)
        dZ = cache[f'A{self.L + 1}'] - Y
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
        AL = cache[f'A{self.L + 1}']  # Output layer activation (softmax probabilities)
        
        # Get the class with highest probability
        predictions = np.argmax(AL, axis=0)
        return predictions.reshape(1, -1)

# Load and preprocess data with make_moons dataset
def load_moons_dataset(n_samples=1000, noise=0.1, test_size=0.2):
    print("Loading make_moons dataset...")
    
    # Generate the moons dataset
    X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Reshape for neural network (features, samples)
    X_train = X_train.T
    X_test = X_test.T
    
    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))
    
    # Transpose to match network's expected shape
    y_train_one_hot = y_train_one_hot.T
    y_test_one_hot = y_test_one_hot.T
    
    # Print shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train_one_hot.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test_one_hot.shape}")
    
    # Feature names for the 2D moons dataset
    feature_names = ['Feature 1', 'Feature 2']
    target_names = ['Class 0', 'Class 1']
    
    return X_train, X_test, y_train_one_hot, y_test_one_hot, y_train, y_test, feature_names, target_names

# Calculate accuracy
def calculate_accuracy(predictions, y):
    return np.mean(predictions == y) * 100

# Visualize decision boundaries
def plot_decision_boundary(X, y, model, feature_names):
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[0].min() - 1, X[0].max() + 1
    y_min, y_max = X[1].min() - 1, X[1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))
    
    # Create input array for prediction
    Z_input = np.zeros((X.shape[0], xx.ravel().shape[0]))
    Z_input[0] = xx.ravel()
    Z_input[1] = yy.ravel()
    
    # Predict class for each point in mesh
    Z = model.predict(Z_input)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    
    # Plot training points
    scatter = plt.scatter(X[0], X[1], c=y, edgecolors='k', marker='o', cmap=plt.cm.Spectral)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Decision Boundary for Make Moons Dataset")
    plt.colorbar(scatter)
    plt.show()
    
def main():
    print("Neural Network with Adam Optimization, Batch Normalization, and Regularization")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load make_moons dataset with some noise
    X_train, X_test, y_train_one_hot, y_test_one_hot, y_train, y_test, feature_names, target_names = load_moons_dataset(
        n_samples=1000, 
        noise=0.1, 
        test_size=0.2
    )
    
    # Initialize the neural network
    input_size = X_train.shape[0]  # 2 features for make_moons
    hidden_layers = [128, 64, 32, 16]  # Smaller network for simple dataset
    output_size = y_train_one_hot.shape[0]  # 2 classes for make_moons
    learning_rate = 0.01
    lambda_reg = 0.01  # L2 regularization parameter
    keep_prob = 0.8    # Dropout keep probability
    use_batch_norm = True  # Enable batch normalization
    
    # Create neural network
    nn = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate, lambda_reg, 
                      keep_prob, beta1=0.9, beta2=0.999, epsilon=1e-8, use_batch_norm=use_batch_norm)
    
    # Train the network
    print("\nTraining neural network...")
    nn.train(X_train, y_train_one_hot, epochs=500, batch_size=32, print_loss=True)
    
    # Evaluate the model
    train_predictions = nn.predict(X_train)
    test_predictions = nn.predict(X_test)
    
    train_accuracy = calculate_accuracy(train_predictions, y_train)
    test_accuracy = calculate_accuracy(test_predictions, y_test)
    
    print(f"\nTraining accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Plot decision boundary
    print("\nPlotting decision boundary...")
    plot_decision_boundary(X_train, y_train, nn, feature_names)
    
    return nn

if __name__ == "__main__":
    # Run the neural network on make_moons dataset
    nn = main()