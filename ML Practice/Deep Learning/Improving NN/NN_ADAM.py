import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, 
                 lambda_reg=0.01, keep_prob=0.8, beta1=0.9, beta2=0.999, epsilon=1e-8):
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
        
        # Initialize weights and biases for all layers
        # layer_dims contains sizes of all layers including input and output
        self.layer_dims = [input_size] + hidden_layers + [output_size]
        
        for l in range(1, len(self.layer_dims)):
            # He initialization for better gradient flow in deep networks
            # W shape: (current_layer_size, previous_layer_size)
            self.parameters[f"W{l}"] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2. / self.layer_dims[l-1])
            # Initialize biases to zeros, shape: (current_layer_size, 1)
            self.parameters[f"b{l}"] = np.zeros((self.layer_dims[l], 1))
        
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

    def forward_propagation(self, X):
        # Cache stores all activations and Z values for backpropagation
        cache = {'A0': X}  # Input layer activation
        A_prev = X  # Current activation to pass forward
        
        # Loop through hidden layers (1 to self.L)
        for l in range(1, self.L + 1): 
            W = self.parameters[f'W{l}']  # Weights of current layer
            b = self.parameters[f'b{l}']  # Biases of current layer
            # Linear transformation: Z = W·A + b
            Z = np.dot(W, A_prev) + b
            # Apply ReLU activation to hidden layers
            A = self.relu(Z)
            
            # Apply dropout only during training
            if self.is_training and l < self.L + 1:  # Apply dropout to all layers except output
                # Create dropout mask (1: keep, 0: drop)
                cache[f'D{l}'] = np.random.rand(*A.shape) < self.keep_prob
                # Apply dropout mask and scale by keep_prob to maintain expected values
                A = A * cache[f'D{l}'] / self.keep_prob
            
            # Store values in cache for backpropagation
            cache[f'Z{l}'] = Z  # Pre-activation
            cache[f'A{l}'] = A  # Post-activation (with dropout if applicable)
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
            
            # Calculate gradients for weights with L2 regularization
            grads[f'dW{l}'] = (np.dot(dZ, cache[f'A{l - 1}'].T) / m) + ((self.lambda_reg / m) * self.parameters[f'W{l}'])
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m

        return grads

    def update_parameters(self, grads):
        # Update all weights and biases using Adam optimization
        # Increment time step for bias correction
        self.t += 1
        
        for l in range(1, self.L + 2):  # +2 because we include output layer
            # Update first moment (momentum) cache
            self.m_cache[f'dW{l}'] = self.beta1 * self.m_cache[f'dW{l}'] + (1 - self.beta1) * grads[f'dW{l}']
            self.m_cache[f'db{l}'] = self.beta1 * self.m_cache[f'db{l}'] + (1 - self.beta1) * grads[f'db{l}']
            
            # Update second moment (velocity) cache
            self.v_cache[f'dW{l}'] = self.beta2 * self.v_cache[f'dW{l}'] + (1 - self.beta2) * np.square(grads[f'dW{l}'])
            self.v_cache[f'db{l}'] = self.beta2 * self.v_cache[f'db{l}'] + (1 - self.beta2) * np.square(grads[f'db{l}'])
            
            # Bias correction for first and second moments
            m_hat_W = self.m_cache[f'dW{l}'] / (1 - self.beta1**self.t)
            m_hat_b = self.m_cache[f'db{l}'] / (1 - self.beta1**self.t)
            v_hat_W = self.v_cache[f'dW{l}'] / (1 - self.beta2**self.t)
            v_hat_b = self.v_cache[f'db{l}'] / (1 - self.beta2**self.t)
            
            # Update parameters using Adam formula
            self.parameters[f'W{l}'] -= self.learning_rate * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
            self.parameters[f'b{l}'] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

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

                # Step 4: Update parameters using RMSprop
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

# Load and preprocess data from sklearn
def load_and_preprocess_data(dataset_name="iris", test_size=0.2):
    print(f"Loading {dataset_name} dataset...")
    
    # Load dataset
    if dataset_name == "iris":
        data = datasets.load_iris()
    elif dataset_name == "wine":
        data = datasets.load_wine()
    elif dataset_name == "breast_cancer":
        data = datasets.load_breast_cancer()
    else:
        # Default to a simple dataset
        data = datasets.load_iris()
        
    X = data.data
    y = data.target
    
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
    
    return X_train, X_test, y_train_one_hot, y_test_one_hot, y_train, y_test, data.feature_names, data.target_names

# Calculate accuracy
def calculate_accuracy(predictions, y):
    return np.mean(predictions == y) * 100

# Visualize decision boundaries
def plot_decision_boundary(X, y, model, feature_names):
    # Only plot for 2D data
    if X.shape[0] > 2:
        # Use first two features
        X_reduced = X[:2, :]
        print("Using only first two features for visualization")
    else:
        X_reduced = X
    
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X_reduced[0].min() - 1, X_reduced[0].max() + 1
    y_min, y_max = X_reduced[1].min() - 1, X_reduced[1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create input array for prediction
    if X.shape[0] > 2:
        # Pad with zeros for additional features
        Z_input = np.zeros((X.shape[0], xx.ravel().shape[0]))
        Z_input[0] = xx.ravel()
        Z_input[1] = yy.ravel()
    else:
        Z_input = np.c_[xx.ravel(), yy.ravel()].T
    
    # Predict class for each point in mesh
    Z = model.predict(Z_input)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8)
    
    # Plot training points
    plt.scatter(X_reduced[0], X_reduced[1], c=y, edgecolors='k', marker='o')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Decision Boundary")
    plt.colorbar()
    plt.show()
    
def main():
    print("Neural Network with RMSprop Optimization and Regularization")
    print("=" * 50)
    
    # Choose dataset
    dataset_name = "breast_cancer"  # Options: "iris", "wine", "breast_cancer"
    
    # Load data
    X_train, X_test, y_train_one_hot, y_test_one_hot, y_train, y_test, feature_names, target_names = load_and_preprocess_data(dataset_name)
    
    # Initialize the neural network
    input_size = X_train.shape[0]
    hidden_layers = [128, 64]  # Smaller network for simpler datasets
    output_size = y_train_one_hot.shape[0]
    learning_rate = 0.01
    lambda_reg = 0.01  # L2 regularization parameter
    keep_prob = 0.8    # Dropout keep probability
    
    # Create neural network
    nn = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate, lambda_reg, keep_prob, beta1=0.9, beta2=0.999, epsilon=1e-8)
    
    # Train the network
    print("\nTraining neural network...")
    nn.train(X_train, y_train_one_hot, epochs=500, batch_size=16, print_loss=True)
    
    # Evaluate the model
    train_predictions = nn.predict(X_train)
    test_predictions = nn.predict(X_test)
    
    train_accuracy = calculate_accuracy(train_predictions, y_train)
    test_accuracy = calculate_accuracy(test_predictions, y_test)
    
    print(f"\nTraining accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Plot decision boundary
    plot_decision_boundary(X_train, y_train, nn, feature_names)
    
    return nn

if __name__ == "__main__":
    # Choose what to run
    nn = main()