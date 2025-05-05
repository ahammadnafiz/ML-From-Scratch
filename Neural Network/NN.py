import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate = 0.01):
        # Store the number of hidden layers
        self.L = len(hidden_layers)
        # Dictionary to store weights and biases
        self.parameters = {}
        self.learning_rate = learning_rate

        # Initialize weights and biases for all layers
        # layer_dims contains sizes of all layers including input and output
        layer_dims = [input_size] + hidden_layers + [output_size]

        for l in range(1, len(layer_dims)):
            # He initialization for better gradient flow in deep networks
            # W shape: (current_layer_size, previous_layer_size)
            self.parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
            # Initialize biases to zeros, shape: (current_layer_size, 1)
            self.parameters[f"b{l}"] = np.zeros((layer_dims[l], 1)) * np.sqrt(2./layer_dims[l-1])

    def relu(self, Z):
        # ReLU activation function: max(0, Z)
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        # Derivative of ReLU: 1 if Z > 0, else 0
        return np.where(Z > 0, 1, 0)

    def softmax(self, Z):
        # Softmax activation for output layer (multi-class classification)
        # Subtracting max(Z) for numerical stability to prevent overflow
        exp_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
        # Normalize to get probability distribution
        return exp_Z / exp_Z.sum(axis = 0, keepdims = True)

    def forward_propagation(self, X):
        # Cache stores all activations and Z values for backpropagation
        cache = {'A0': X} # Input layer activation
        A_prev = X # Current activation to pass forward
        
        # Loop through hidden layers (1 to self.L)
        for l in range(1, self.L + 1): 
            W = self.parameters[f'W{l}']  # Weights of current layer
            b = self.parameters[f'b{l}']  # Biases of current layer
            # Linear transformation: Z = W·A + b
            Z = np.dot(W, A_prev) + b
            # Apply ReLU activation to hidden layers
            A = self.relu(Z)
            
            # Store values in cache for backpropagation
            cache[f'Z{l}'] = Z  # Pre-activation
            cache[f'A{l}'] = A  # Post-activation
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
        return -np.sum(Y * np.log(AL + 1e-8)) / m

    def backward_propagation(self, cache, Y):
        grads = {}  # Store gradients for all parameters
        m = Y.shape[1]  # Number of examples

        # Output Layer gradient calculation
        # For softmax with cross-entropy, gradient is (prediction - actual)
        dZ = cache[f'A{self.L + 1}'] - Y
        # Gradient of weights: dW = (dZ · previous_activation_transpose) / m
        grads[f'dW{self.L + 1}'] = np.dot(dZ, cache[f'A{self.L}'].T) / m
        # Gradient of biases: sum dZ across examples
        grads[f'db{self.L + 1}'] = np.sum(dZ, axis = 1, keepdims = True) / m
        
        # Hidden Layers gradient calculation (backpropagating the error)
        for l in range(self.L, 0, -1):  # Loop backwards through hidden layers
            # Propagate error from layer l+1 to layer l
            dA_prev = np.dot(self.parameters[f'W{l + 1}'].T, dZ)
            # Apply ReLU derivative to get dZ for current layer
            dZ = dA_prev * self.relu_derivative(cache[f'Z{l}'])
            
            # Calculate gradients for weights and biases
            grads[f'dW{l}'] = np.dot(dZ, cache[f'A{l - 1}'].T) / m
            grads[f'db{l}'] = np.sum(dZ, axis = 1, keepdims = True) / m

        return grads

    def update_parameters(self, grads):
        # Update all weights and biases using gradient descent
        for l in range(1, self.L + 2):  # +2 because we include output layer
            # W = W - learning_rate * dW
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            # b = b - learning_rate * db
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']

    def train(self, X_train, Y_train, epochs = 1000, print_loss = True):
        for epoch in range(epochs):
            # Step 1: Forward propagation - compute activations
            cache = self.forward_propagation(X_train)
            AL = cache[f'A{self.L + 1}']  # Output layer activation

            # Step 2: Compute loss
            loss = self.compute_loss(AL, Y_train)

            # Step 3: Backward propagation - compute gradients
            grads = self.backward_propagation(cache, Y_train)

            # Step 4: Update parameters using gradients
            self.update_parameters(grads)

            # Print loss every 100 epochs if requested
            if print_loss and epoch % 100 == 0:
                print(f'Epoch {epoch} loss: {loss:.4f}')

    def predict(self, X):
        # Forward pass to get output probabilities
        cache = self.forward_propagation(X)
        AL = cache[f'A{self.L + 1}']  # Output layer activation (softmax probabilities)
        
        # Get the class with highest probability
        predictions = np.argmax(AL, axis = 0)
        return predictions.reshape(1, -1)

# Load MNIST data
def load_mnist_data(samples=5000):
    print("Loading MNIST data...")
    # Fetch the MNIST dataset from OpenML
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    
    # Normalize pixel values to range [0,1]
    X = X.astype('float32') / 255.0
    
    # Use only a subset of data to speed up training
    if samples is not None:
        X = X[:samples]
        y = y[:samples]
    
    return X, y

# Preprocess data
def preprocess_data(X, y, test_size=0.2):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Reshape X for the neural network input format (features, samples)
    # Neural network expects inputs as (784, m) where m is number of examples
    X_train = X_train.T
    X_test = X_test.T
    
    # Convert pandas Series to numpy arrays if needed
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    # One-hot encode the labels (convert digits to binary vectors)
    encoder = OneHotEncoder(sparse_output=False)
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))
    
    # Transpose to shape (10, m) to match neural network output
    y_train_one_hot = y_train_one_hot.T
    y_test_one_hot = y_test_one_hot.T
    
    # Print shapes for debugging
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train_one_hot.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test_one_hot.shape}")
    
    return X_train, X_test, y_train_one_hot, y_test_one_hot, y_train, y_test

# Calculate accuracy
def calculate_accuracy(predictions, y):
    # Percentage of correct predictions
    return np.mean(predictions == y) * 100

# Visualize some predictions
def visualize_predictions(X, predictions, actual, num_images=5):
    plt.figure(figsize=(15, 3))
    
    # Convert X to numpy array if it's a pandas DataFrame
    if hasattr(X, 'values'):
        X_array = X.values
    else:
        X_array = X
        
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        
        # Get the image data and reshape it
        if X_array.shape[0] == 784:  # If X is transposed (features, samples)
            img_data = X_array[:, i]
        else:  # If X is in standard format (samples, features)
            img_data = X_array[i, :]
            
        # Reshape 784 pixel values to 28x28 image and display
        plt.imshow(img_data.reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {predictions[0, i]}\nActual: {actual[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Load MNIST data (using a small subset for faster training)
    X, y = load_mnist_data(samples=5000)
    
    # Preprocess the data - split, reshape, one-hot encode
    X_train, X_test, y_train_one_hot, y_test_one_hot, y_train, y_test = preprocess_data(X, y)
    
    # Initialize the neural network
    input_size = 784  # 28x28 pixels flattened
    hidden_layers = [128, 64]  # Two hidden layers with 128 and 64 neurons
    output_size = 10  # 10 digits (0-9)
    learning_rate = 0.1
    
    # Create neural network instance
    nn = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate)
    
    # Train the neural network
    print("Training neural network...")
    nn.train(X_train, y_train_one_hot, epochs=1000, print_loss=True)
    
    # Make predictions on training and test sets
    print("Making predictions...")
    train_predictions = nn.predict(X_train)
    test_predictions = nn.predict(X_test)
    
    # Calculate and display accuracy
    train_accuracy = calculate_accuracy(train_predictions, y_train.astype(int))
    test_accuracy = calculate_accuracy(test_predictions, y_test.astype(int))
    
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Visualize some test predictions
    print("Visualizing some predictions...")
    visualize_predictions(X_test, test_predictions, y_test, num_images=5)
    
    return nn

if __name__ == "__main__":
    nn = main()