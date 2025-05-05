import numpy as np
import matplotlib.pyplot as plt

# Dimensions
n_x = 4  # Input size
n_h = 5  # Hidden layer size
n_y = 3  # Output size
T = 3    # Number of time steps

# Random seed for reproducibility
np.random.seed(0)

# Parameters initialization
W_hx = np.random.randn(n_h, n_x) * 0.01  # Input to hidden layer weights
W_hh = np.random.randn(n_h, n_h) * 0.01  # Hidden to hidden layer weights
W_yh = np.random.randn(n_y, n_h) * 0.01  # Hidden to output layer weights

b_h = np.zeros((n_h, 1))  # Hidden layer bias
b_y = np.zeros((n_y, 1))  # Output layer bias

# Activation functions
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - np.tanh(x) ** 2

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def rnn_forward(X, Y):
    h = {-1: np.zeros((n_h, 1))} # Initial hidden state
    a, z, y_hat = {}, {}, {}
    loss = 0

    for t in range(T):
        x_t = X[t].reshape(-1, 1)
        a[t] = np.dot(W_hx, x_t) + np.dot(W_hh, h[t-1]) + b_h
        h[t] = np.tanh(a[t])
        z[t] = np.dot(W_yh, h[t]) + b_y
        y_hat[t] = softmax(z[t])
        loss += -np.sum(Y[t].reshape(-1, 1) * np.log(y_hat[t] + 1e-8))  # Avoid log(0)

    cache = (X, Y, a, h, z, y_hat)
    return loss, cache

def rnn_backward(cache):
    X, Y, a, h, z, y_hat = cache

    dW_hx = np.zeros_like(W_hx)
    dW_hh = np.zeros_like(W_hh)
    db_h = np.zeros_like(b_h)
    dW_yh = np.zeros_like(W_yh)
    db_y = np.zeros_like(b_y)

    dh_next = np.zeros_like(h[0])
    
    for t in reversed(range(T)):
        dz = y_hat[t] - Y[t].reshape(-1, 1)  # Gradient of loss w.r.t. z
        dW_yh += np.dot(dz, h[t].T)
        db_y += dz
        
        dh = np.dot(W_yh.T, dz) + dh_next  # Gradient of loss w.r.t. h
        da = dtanh(a[t]) * dh
        dW_hh += np.dot(da, h[t-1].T)
        dW_hx += np.dot(da, X[t].reshape(-1, 1).T)
        db_h += da
        dh_next = np.dot(W_hh.T, da)
        dh_next = dh_next * (1 - np.tanh(a[t]) ** 2)
        # Note: We do not need to backpropagate through the input layer
        # as we are not updating W_xh in this example.
    # Clip gradients to prevent exploding gradients
    for dparam in [dW_hx, dW_hh, db_h, dW_yh, db_y]:
        np.clip(dparam, -5, 5, out=dparam)
        
    gradients = {
        'dW_hx': dW_hx,
        'dW_hh': dW_hh,
        'db_h': db_h,
        'dW_yh': dW_yh,
        'db_y': db_y
    }
    
    return gradients

def update_parameters(gradients, learning_rate=0.01):
    """
    Update the parameters using gradient descent
    """
    global W_hx, W_hh, b_h, W_yh, b_y
    
    # Update parameters using gradient descent
    W_hx -= learning_rate * gradients['dW_hx']
    W_hh -= learning_rate * gradients['dW_hh']
    b_h -= learning_rate * gradients['db_h']
    W_yh -= learning_rate * gradients['dW_yh']
    b_y -= learning_rate * gradients['db_y']

def predict(X):
    """
    Predict function for making predictions
    """
    h = {}
    h[-1] = np.zeros((n_h, 1))
    y_hat = {}
    
    # Forward pass only
    for t in range(len(X)):
        x_t = X[t].reshape(-1, 1)
        h[t] = tanh(np.dot(W_hx, x_t) + np.dot(W_hh, h[t-1]) + b_h)
        z_t = np.dot(W_yh, h[t]) + b_y
        y_hat[t] = softmax(z_t)
    
    # Return predictions (class indices)
    predictions = [np.argmax(y_hat[t]) for t in range(len(X))]
    return predictions

# Generate toy data examples
def generate_toy_data(num_samples, sequence_length=T):
    """
    Generate toy data for training and testing
    Simple pattern: Sum of inputs determines the class
    """
    X_data = []
    Y_data = []
    
    for _ in range(num_samples):
        # Create a random sequence
        sequence = [np.random.randn(n_x) for _ in range(sequence_length)]
        X_data.append(sequence)
        
        # Create target labels (example rule: sum of inputs determines class)
        targets = []
        for t in range(sequence_length):
            # Sum of input values > 0 => class 0
            # Sum of input values < 0 => class 1
            # Otherwise => class 2
            input_sum = np.sum(sequence[t])
            if input_sum > 0.5:
                target = 0
            elif input_sum < -0.5:
                target = 1
            else:
                target = 2
                
            # One-hot encode
            one_hot = np.zeros(n_y)
            one_hot[target] = 1
            targets.append(one_hot)
            
        Y_data.append(targets)
    
    return X_data, Y_data

# Train the RNN model
def train_rnn(X_data, Y_data, num_epochs=100, learning_rate=0.01):
    """
    Train the RNN model with the provided data
    """
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for i in range(len(X_data)):
            X = X_data[i]
            Y = Y_data[i]
            
            # Forward pass
            loss, cache = rnn_forward(X, Y)
            epoch_loss += loss
            
            # Backward pass
            gradients = rnn_backward(cache)
            
            # Update parameters
            update_parameters(gradients, learning_rate)
        
        # Track progress
        avg_loss = epoch_loss / len(X_data)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    return losses

# Evaluate the model
def evaluate_model(X_test, Y_test):
    """
    Evaluate the model's performance
    """
    correct = 0
    total = 0
    
    for i in range(len(X_test)):
        X = X_test[i]
        Y = Y_test[i]
        
        # Get predictions
        predictions = predict(X)
        
        # Get true labels
        true_labels = [np.argmax(Y[t]) for t in range(len(Y))]
        
        # Count correct predictions
        for t in range(len(predictions)):
            if predictions[t] == true_labels[t]:
                correct += 1
            total += 1
    
    accuracy = correct / total
    return accuracy

# Run the example
if __name__ == "__main__":
    print("Generating toy data...")
    num_samples = 100
    X_train, Y_train = generate_toy_data(num_samples)
    X_test, Y_test = generate_toy_data(20)
    
    print("Training RNN model...")
    losses = train_rnn(X_train, Y_train, num_epochs=100, learning_rate=0.01)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('rnn_training_loss.png')
    plt.close()
    
    # Evaluate on test data
    accuracy = evaluate_model(X_test, Y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Run a single example to demonstrate prediction
    sample_X = X_test[0]
    sample_Y = Y_test[0]
    
    predictions = predict(sample_X)
    true_labels = [np.argmax(sample_Y[t]) for t in range(len(sample_Y))]
    
    print("\nSample Prediction Example:")
    for t in range(len(predictions)):
        print(f"Time step {t}:")
        print(f"  Input: {sample_X[t]}")
        print(f"  Predicted class: {predictions[t]}")
        print(f"  True class: {true_labels[t]}")