import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

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
                print(f'Epoch {epoch} loss: {loss:.4f}')

    def predict(self, X):
        cache = self.forward_propagation(X)
        AL = cache[f'A{self.L + 1}'] # Output layer activation (softmax probabilities)
        
        predictions = np.argmax(AL, axis = 0)
        return predictions.reshape(1, -1)

# Load MNIST data
def load_mnist_data(samples=5000):
    print("Loading MNIST data...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    
    # Convert to float32 for better efficiency
    X = X.astype('float32') / 255.0
    
    # Take a subset for faster training
    if samples is not None:
        X = X[:samples]
        y = y[:samples]
    
    return X, y

# Preprocess data
def preprocess_data(X, y, test_size=0.2):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Reshape X for the neural network (features, samples)
    X_train = X_train.T
    X_test = X_test.T
    
    # Convert pandas Series to numpy arrays if needed
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))
    
    y_train_one_hot = y_train_one_hot.T
    y_test_one_hot = y_test_one_hot.T
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train_one_hot.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test_one_hot.shape}")
    
    return X_train, X_test, y_train_one_hot, y_test_one_hot, y_train, y_test

# Calculate accuracy
def calculate_accuracy(predictions, y):
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
            
        plt.imshow(img_data.reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {predictions[0, i]}\nActual: {actual[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Load MNIST data (using a small subset for faster training)
    X, y = load_mnist_data(samples=5000)
    
    # Preprocess the data
    X_train, X_test, y_train_one_hot, y_test_one_hot, y_train, y_test = preprocess_data(X, y)
    
    # Initialize the neural network
    input_size = 784  # 28x28 pixels
    hidden_layers = [128, 64]  # Two hidden layers
    output_size = 10  # 10 digits (0-9)
    learning_rate = 0.1
    
    nn = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate)
    
    # Train the neural network
    print("Training neural network...")
    nn.train(X_train, y_train_one_hot, epochs=1000, print_loss=True)
    
    # Make predictions
    print("Making predictions...")
    train_predictions = nn.predict(X_train)
    test_predictions = nn.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = calculate_accuracy(train_predictions, y_train.astype(int))
    test_accuracy = calculate_accuracy(test_predictions, y_test.astype(int))
    
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Visualize some predictions
    print("Visualizing some predictions...")
    visualize_predictions(X_test, test_predictions, y_test, num_images=5)
    
    return nn

if __name__ == "__main__":
    nn = main()