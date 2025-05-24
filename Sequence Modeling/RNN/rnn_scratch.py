import numpy as np
import matplotlib.pyplot as plt

# Read and preprocess text data
def load_text_data(filename):
    """Load text data and create character mappings"""
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Get unique characters and create mappings
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Text length: {len(text)} characters")
    print(f"Vocabulary size: {len(chars)} unique characters")
    print(f"First 10 characters: {chars[:10]}")
    
    return text, chars, char_to_idx, idx_to_char

# Load the text data
text, chars, char_to_idx, idx_to_char = load_text_data('input.txt')

# Dimensions
n_x = len(chars)  # Input size (vocabulary size)
n_h = 50          # Hidden layer size (reduced for faster training)
n_y = len(chars)  # Output size (vocabulary size)
T = 15            # Sequence length for training (reduced for faster training)

# Random seed for reproducibility
np.random.seed(0)

# Parameters initialization
# Initialize with small random values to break symmetry
W_hx = np.random.randn(n_h, n_x) * 0.01  # Input to hidden layer weights
W_hh = np.random.randn(n_h, n_h) * 0.01  # Hidden to hidden layer weights (recurrent connections)
W_yh = np.random.randn(n_y, n_h) * 0.01  # Hidden to output layer weights

b_h = np.zeros((n_h, 1))  # Hidden layer bias
b_y = np.zeros((n_y, 1))  # Output layer bias

# Activation functions
def tanh(x):
    # Hyperbolic tangent: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    # Range: [-1, 1]
    return np.tanh(x)

def dtanh(x):
    # Derivative of tanh: d/dx tanh(x) = 1 - tanh(x)^2
    return 1.0 - np.tanh(x) ** 2

def softmax(x):
    # Softmax function: σ(z)_j = e^(z_j) / Σ_k(e^(z_k))
    # Subtracting max(x) for numerical stability to avoid overflow
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def rnn_forward(X, Y):
    # Forward pass through the RNN
    h = {-1: np.zeros((n_h, 1))}  # Initial hidden state h₀ = 0
    a, z, y_hat = {}, {}, {}
    loss = 0

    for t in range(T):
        x_t = X[t].reshape(-1, 1)  # Input at time t
        
        # Hidden state update: a_t = W_hx * x_t + W_hh * h_(t-1) + b_h
        a[t] = np.dot(W_hx, x_t) + np.dot(W_hh, h[t-1]) + b_h
        
        # Apply tanh activation: h_t = tanh(a_t)
        h[t] = np.tanh(a[t])
        
        # Output layer: z_t = W_yh * h_t + b_y
        z[t] = np.dot(W_yh, h[t]) + b_y
        
        # Apply softmax for probability distribution: y_hat_t = softmax(z_t)
        y_hat[t] = softmax(z[t])
        
        # Cross-entropy loss: L_t = -Σ(y_t * log(y_hat_t))
        # 1e-8 added to avoid log(0)
        loss += -np.sum(Y[t].reshape(-1, 1) * np.log(y_hat[t] + 1e-8))

    # Save values for backpropagation
    cache = (X, Y, a, h, z, y_hat)
    return loss, cache

def rnn_backward(cache):
    # Backpropagation through time (BPTT)
    X, Y, a, h, z, y_hat = cache

    # Initialize gradient accumulators
    dW_hx = np.zeros_like(W_hx)
    dW_hh = np.zeros_like(W_hh)
    db_h = np.zeros_like(b_h)
    dW_yh = np.zeros_like(W_yh)
    db_y = np.zeros_like(b_y)

    # Initialize gradient of future hidden state
    dh_next = np.zeros_like(h[0])
    
    # Iterate backwards through time
    for t in reversed(range(T)):
        # Gradient of loss w.r.t softmax output: dL/dz_t = y_hat_t - y_t
        # This is the derivative of cross-entropy loss with softmax
        dz = y_hat[t] - Y[t].reshape(-1, 1)
        
        # Gradient w.r.t W_yh: dL/dW_yh = dL/dz_t * h_t^T
        dW_yh += np.dot(dz, h[t].T)
        
        # Gradient w.r.t b_y: dL/db_y = dL/dz_t
        db_y += dz
        
        # Gradient w.r.t hidden state: dL/dh_t = W_yh^T * dL/dz_t + dL/dh_(t+1)
        # The second term comes from the recurrent connection (future gradient)
        dh = np.dot(W_yh.T, dz) + dh_next
        
        # Gradient w.r.t pre-activation: dL/da_t = dL/dh_t * tanh'(a_t)
        da = dtanh(a[t]) * dh
        
        # Gradient w.r.t W_hh: dL/dW_hh = dL/da_t * h_(t-1)^T
        dW_hh += np.dot(da, h[t-1].T)
        
        # Gradient w.r.t W_hx: dL/dW_hx = dL/da_t * x_t^T
        dW_hx += np.dot(da, X[t].reshape(-1, 1).T)
        
        # Gradient w.r.t b_h: dL/db_h = dL/da_t
        db_h += da
        
        # Gradient for next iteration: dL/dh_(t-1) = W_hh^T * dL/da_t
        dh_next = np.dot(W_hh.T, da)
        
    # Clip gradients to prevent exploding gradients problem
    # This is a common technique in RNN training
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
    Update the parameters using gradient descent:
    θ = θ - α * ∇J(θ)
    where:
    θ: parameter
    α: learning rate
    ∇J(θ): gradient of cost function w.r.t parameter
    """
    global W_hx, W_hh, b_h, W_yh, b_y
    
    # Update each parameter by subtracting the scaled gradient
    W_hx -= learning_rate * gradients['dW_hx']
    W_hh -= learning_rate * gradients['dW_hh']
    b_h -= learning_rate * gradients['db_h']
    W_yh -= learning_rate * gradients['dW_yh']
    b_y -= learning_rate * gradients['db_y']

def predict(X):
    """
    Make predictions using the trained RNN model
    Only performs the forward pass without computing loss
    """
    h = {}
    h[-1] = np.zeros((n_h, 1))  # Initialize first hidden state to zeros
    y_hat = {}
    
    # Forward pass only
    for t in range(len(X)):
        x_t = X[t].reshape(-1, 1)
        # h_t = tanh(W_hx * x_t + W_hh * h_(t-1) + b_h)
        h[t] = tanh(np.dot(W_hx, x_t) + np.dot(W_hh, h[t-1]) + b_h)
        # z_t = W_yh * h_t + b_y
        z_t = np.dot(W_yh, h[t]) + b_y
        # y_hat_t = softmax(z_t)
        y_hat[t] = softmax(z_t)
    
    # Return class with highest probability for each time step
    predictions = [np.argmax(y_hat[t]) for t in range(len(X))]
    return predictions

# Generate character-level training data
def generate_char_data(text, char_to_idx, sequence_length=T):
    """
    Generate character-level training sequences from text
    Each sequence predicts the next character
    """
    X_data = []
    Y_data = []
    
    # Create sequences of length T
    for i in range(0, len(text) - sequence_length, sequence_length):
        # Input sequence
        input_seq = text[i:i + sequence_length]
        # Target sequence (shifted by 1)
        target_seq = text[i + 1:i + sequence_length + 1]
        
        # Convert to one-hot encoding
        X_seq = []
        Y_seq = []
        
        for char in input_seq:
            # One-hot encode input character
            one_hot_input = np.zeros(n_x)
            one_hot_input[char_to_idx[char]] = 1
            X_seq.append(one_hot_input)
        
        for char in target_seq:
            # One-hot encode target character
            one_hot_target = np.zeros(n_y)
            one_hot_target[char_to_idx[char]] = 1
            Y_seq.append(one_hot_target)
        
        X_data.append(X_seq)
        Y_data.append(Y_seq)
    
    return X_data, Y_data

# Train the RNN model
def train_rnn(X_data, Y_data, num_epochs=100, learning_rate=0.01):
    """
    Train the RNN model with the provided data using gradient descent
    """
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for i in range(len(X_data)):
            X = X_data[i]
            Y = Y_data[i]
            
            # Forward pass: compute loss L(θ)
            loss, cache = rnn_forward(X, Y)
            epoch_loss += loss
            
            # Backward pass: compute gradients ∇L(θ)
            gradients = rnn_backward(cache)
            
            # Update parameters: θ = θ - α * ∇L(θ)
            update_parameters(gradients, learning_rate)
        
        # Track average loss per epoch
        avg_loss = epoch_loss / len(X_data)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    return losses

def generate_text(seed_text, length=100, temperature=1.0):
    """
    Generate text using the trained RNN model
    
    Args:
        seed_text: Starting text to prime the generation
        length: Number of characters to generate
        temperature: Controls randomness (lower = more conservative, higher = more random)
    """
    # Convert seed text to indices
    generated = seed_text
    
    # Initialize hidden state
    h = np.zeros((n_h, 1))
    
    # Process seed text to warm up the hidden state
    for char in seed_text:
        if char in char_to_idx:
            x = np.zeros((n_x, 1))
            x[char_to_idx[char], 0] = 1
            
            # Forward pass
            a = np.dot(W_hx, x) + np.dot(W_hh, h) + b_h
            h = np.tanh(a)
    
    # Generate new characters
    for _ in range(length):
        # Get last character
        last_char = generated[-1] if generated else ' '
        
        if last_char in char_to_idx:
            x = np.zeros((n_x, 1))
            x[char_to_idx[last_char], 0] = 1
            
            # Forward pass
            a = np.dot(W_hx, x) + np.dot(W_hh, h) + b_h
            h = np.tanh(a)
            z = np.dot(W_yh, h) + b_y
            
            # Apply temperature scaling
            z = z / temperature
            y_hat = softmax(z)
            
            # Sample from the probability distribution
            probabilities = y_hat.flatten()
            char_idx = np.random.choice(len(chars), p=probabilities)
            next_char = idx_to_char[char_idx]
            
            generated += next_char
        else:
            # If character not in vocabulary, add a space
            generated += ' '
    
    return generated

# Evaluate the model for text generation
def evaluate_text_model(X_test, Y_test):
    """
    Evaluate the model's performance on character prediction
    """
    correct = 0
    total = 0
    
    for i in range(min(len(X_test), 50)):  # Evaluate on subset for speed
        X = X_test[i]
        Y = Y_test[i]
        
        # Get model predictions
        predictions = predict(X)
        
        # Get true labels from one-hot encoded targets
        true_labels = [np.argmax(Y[t]) for t in range(len(Y))]
        
        # Count correct predictions across all time steps
        for t in range(len(predictions)):
            if predictions[t] == true_labels[t]:
                correct += 1
            total += 1
    
    # Accuracy = correct / total
    accuracy = correct / total if total > 0 else 0
    return accuracy

# Run the example
if __name__ == "__main__":
    print("Generating character-level training data...")
    
    # Split text into train and test
    split_idx = int(0.9 * len(text))
    train_text = text[:split_idx]
    test_text = text[split_idx:]
    
    # Generate training and test data
    X_train, Y_train = generate_char_data(train_text, char_to_idx)
    X_test, Y_test = generate_char_data(test_text, char_to_idx)
    
    print(f"Training sequences: {len(X_train)}")
    print(f"Test sequences: {len(X_test)}")
    
    print("Training RNN model...")
    losses = train_rnn(X_train, Y_train, num_epochs=50, learning_rate=0.001)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss (Character-Level Text Generation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('rnn_text_training_loss.png')
    plt.close()
    
    # Evaluate on test data
    accuracy = evaluate_text_model(X_test, Y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate sample text
    print("\nGenerating sample text...")
    seed_texts = ["The ", "When ", "From "]
    
    for seed in seed_texts:
        print(f"\nSeed: '{seed}'")
        generated = generate_text(seed, length=200, temperature=0.8)
        print(f"Generated: {generated}")
        print("-" * 80)
    
    # Show a prediction example
    if len(X_test) > 0:
        sample_X = X_test[0]
        sample_Y = Y_test[0]
        
        predictions = predict(sample_X)
        true_labels = [np.argmax(sample_Y[t]) for t in range(len(sample_Y))]
        
        print("\nSample Character Prediction Example:")
        input_chars = [idx_to_char[np.argmax(sample_X[t])] for t in range(len(sample_X))]
        predicted_chars = [idx_to_char[predictions[t]] for t in range(len(predictions))]
        true_chars = [idx_to_char[true_labels[t]] for t in range(len(true_labels))]
        
        print(f"Input sequence: {''.join(input_chars)}")
        print(f"Predicted next: {''.join(predicted_chars)}")
        print(f"True next:      {''.join(true_chars)}")