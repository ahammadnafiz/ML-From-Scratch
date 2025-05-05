import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import the neural network code from the uploaded file
from NN_Batch_Nom import NeuralNetwork, calculate_accuracy, plot_decision_boundary

# Define our flower dataset
def create_flower_dataset(n_samples=1500, noise=0.2, test_size=0.2):
    """
    Creates a flower-shaped dataset with 5 petals.
    - Class 1: Points forming a flower pattern with 5 petals
    - Class 0: Points in a ring around the flower and some in the center
    """
    print("Generating flower dataset...")
    
    # Parameters for the flower shape
    petals = 5
    petal_length = 3
    inner_radius = 1
    
    # Arrays to store data
    data = []
    labels = []
    
    # Generate points for the flower class (label 1)
    for i in range(n_samples // 2):
        # Generate random angle and radius
        angle = np.random.uniform(0, 2 * np.pi)
        # Use petal function to create the flower shape: r = a + b*cos(n*theta)
        radius_offset = (np.random.random() - 0.5) * noise * 2  # Add noise
        radius = inner_radius + petal_length * np.abs(np.cos(petals * angle / 2)) + radius_offset
        
        # Convert polar to cartesian coordinates
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        data.append([x, y])
        labels.append(1)
    
    # Generate background points (label 0)
    # Create a ring around the flower
    for i in range(n_samples // 2):
        angle = np.random.uniform(0, 2 * np.pi)
        # Random radius outside and inside the flower
        if np.random.random() < 0.5:
            # Points outside the flower
            radius = inner_radius + petal_length * 1.5 + np.random.random() * 2
        else:
            # Some points inside the inner circle
            radius = np.random.random() * (inner_radius * 0.7)
        
        x = radius * np.cos(angle) + (np.random.random() - 0.5) * noise
        y = radius * np.sin(angle) + (np.random.random() - 0.5) * noise
        
        data.append([x, y])
        labels.append(0)
    
    # Convert to numpy arrays
    X = np.array(data)
    y = np.array(labels)
    
    # Shuffle the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Split into train and test
    split_idx = int((1 - test_size) * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Standardize features (optional - but often helps neural networks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape for neural network (features, samples)
    X_train = X_train.T
    X_test = X_test.T
    
    # One-hot encode labels
    def one_hot_encode(labels, num_classes=2):
        one_hot = np.zeros((num_classes, len(labels)))
        for i, label in enumerate(labels):
            one_hot[label, i] = 1
        return one_hot
    
    y_train_one_hot = one_hot_encode(y_train)
    y_test_one_hot = one_hot_encode(y_test)
    
    # Print shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train_one_hot.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test_one_hot.shape}")
    
    # Feature names for our dataset
    feature_names = ['X-coordinate', 'Y-coordinate']
    target_names = ['Background', 'Flower']
    
    return X_train, X_test, y_train_one_hot, y_test_one_hot, y_train, y_test, feature_names, target_names

# Visualize the raw dataset before training
def visualize_raw_dataset(X, y, feature_names):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[0], X[1], c=y, cmap=plt.cm.Spectral, edgecolors='k', s=40)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Flower Dataset Visualization")
    plt.colorbar()
    plt.grid(alpha=0.3)
    plt.show()

def main():
    print("Testing Neural Network with Custom Flower Dataset")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load our custom flower dataset
    X_train, X_test, y_train_one_hot, y_test_one_hot, y_train, y_test, feature_names, target_names = create_flower_dataset(
        n_samples=1500, 
        noise=0.2, 
        test_size=0.2
    )
    
    # Visualize the dataset before training
    print("\nVisualizing the raw dataset...")
    visualize_raw_dataset(X_train, y_train, feature_names)
    
    # Initialize the neural network
    input_size = X_train.shape[0]  # 2 features
    hidden_layers = [128, 64, 32, 16]  # A bit larger network for the complex boundary
    output_size = y_train_one_hot.shape[0]  # 2 classes
    learning_rate = 0.005
    lambda_reg = 0.01  # L2 regularization parameter
    keep_prob = 0.8    # Dropout keep probability
    use_batch_norm = True  # Enable batch normalization
    
    print("\nInitializing neural network with:")
    print(f"- Input size: {input_size}")
    print(f"- Hidden layers: {hidden_layers}")
    print(f"- Output size: {output_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- L2 regularization: {lambda_reg}")
    print(f"- Dropout keep probability: {keep_prob}")
    print(f"- Batch normalization: {use_batch_norm}")
    
    # Create neural network
    nn = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate, lambda_reg, 
                      keep_prob, beta1=0.9, beta2=0.999, epsilon=1e-8, use_batch_norm=use_batch_norm)
    
    # Train the network
    print("\nTraining neural network...")
    loss_history = nn.train(X_train, y_train_one_hot, epochs=1000, batch_size=32, print_loss=True)
    
    # Plot the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_history)), loss_history)
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
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
    
    # Plot decision boundary for test data as well
    print("\nPlotting decision boundary with test data...")
    plot_decision_boundary(X_test, y_test, nn, feature_names)
    
    return nn

if __name__ == "__main__":
    # Run the neural network on our flower dataset
    nn = main()