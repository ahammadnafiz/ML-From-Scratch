import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import torch

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

# Initialize the embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight 384-dimensional embeddings
embedding_dim = 384  # Dimension of the embedding vectors

# Function to get embeddings for characters/text chunks
def get_text_embeddings(text_chunks):
    """Get embeddings for text chunks using the Hugging Face model"""
    with torch.no_grad():
        embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    return embeddings

# Load the text data - use absolute path to ensure file is found
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
text_file_path = os.path.join(script_dir, 'eng-fra.txt')
text, chars, char_to_idx, idx_to_char = load_text_data(text_file_path)

# Dimensions
n_x = embedding_dim   # Input size (embedding dimension instead of vocabulary size)
n_h = 50             # Hidden layer size
n_y = len(chars)     # Output size (vocabulary size for character prediction)
T = 6                # Sequence length for training (using minimum 6 as requested)

# Random seed for reproducibility
np.random.seed(0)

# Parameters initialization
# Initialize with small random values to break symmetry
W_hx = np.random.randn(n_h, n_x) * 0.01  # Input to hidden layer weights
W_hh = np.random.randn(n_h, n_h) * 0.01  # Hidden to hidden layer weights (recurrent connections)
W_yh = np.random.randn(n_y, n_h) * 0.01  # Hidden to output layer weights

b_h = np.zeros((n_h, 1))  # Hidden layer bias
b_y = np.zeros((n_y, 1))  # Output layer bias

# Adam optimizer parameters
# First moment estimates (momentum)
m_W_hx = np.zeros_like(W_hx)
m_W_hh = np.zeros_like(W_hh)
m_b_h = np.zeros_like(b_h)
m_W_yh = np.zeros_like(W_yh)
m_b_y = np.zeros_like(b_y)

# Second moment estimates (RMSprop)
v_W_hx = np.zeros_like(W_hx)
v_W_hh = np.zeros_like(W_hh)
v_b_h = np.zeros_like(b_h)
v_W_yh = np.zeros_like(W_yh)
v_b_y = np.zeros_like(b_y)

# Adam hyperparameters
beta1 = 0.9      # Exponential decay rate for first moment estimates
beta2 = 0.999    # Exponential decay rate for second moment estimates
epsilon = 1e-8   # Small constant to prevent division by zero
t = 0            # Time step counter for bias correction

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

def update_parameters_adam(gradients, learning_rate=0.001):
    """
    Update parameters using Adam optimizer:
    
    Adam combines momentum (first moment) and RMSprop (second moment):
    m_t = β₁ * m_(t-1) + (1 - β₁) * ∇J(θ)
    v_t = β₂ * v_(t-1) + (1 - β₂) * (∇J(θ))²
    
    With bias correction:
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)
    
    Parameter update:
    θ = θ - α * m̂_t / (√v̂_t + ε)
    """
    global W_hx, W_hh, b_h, W_yh, b_y
    global m_W_hx, m_W_hh, m_b_h, m_W_yh, m_b_y
    global v_W_hx, v_W_hh, v_b_h, v_W_yh, v_b_y
    global t
    
    # Increment time step
    t += 1
    
    # Update first moment (momentum) for each parameter
    m_W_hx = beta1 * m_W_hx + (1 - beta1) * gradients['dW_hx']
    m_W_hh = beta1 * m_W_hh + (1 - beta1) * gradients['dW_hh']
    m_b_h = beta1 * m_b_h + (1 - beta1) * gradients['db_h']
    m_W_yh = beta1 * m_W_yh + (1 - beta1) * gradients['dW_yh']
    m_b_y = beta1 * m_b_y + (1 - beta1) * gradients['db_y']
    
    # Update second moment (RMSprop) for each parameter
    v_W_hx = beta2 * v_W_hx + (1 - beta2) * np.square(gradients['dW_hx'])
    v_W_hh = beta2 * v_W_hh + (1 - beta2) * np.square(gradients['dW_hh'])
    v_b_h = beta2 * v_b_h + (1 - beta2) * np.square(gradients['db_h'])
    v_W_yh = beta2 * v_W_yh + (1 - beta2) * np.square(gradients['dW_yh'])
    v_b_y = beta2 * v_b_y + (1 - beta2) * np.square(gradients['db_y'])
    
    # Bias correction
    m_W_hx_corrected = m_W_hx / (1 - beta1**t)
    m_W_hh_corrected = m_W_hh / (1 - beta1**t)
    m_b_h_corrected = m_b_h / (1 - beta1**t)
    m_W_yh_corrected = m_W_yh / (1 - beta1**t)
    m_b_y_corrected = m_b_y / (1 - beta1**t)
    
    v_W_hx_corrected = v_W_hx / (1 - beta2**t)
    v_W_hh_corrected = v_W_hh / (1 - beta2**t)
    v_b_h_corrected = v_b_h / (1 - beta2**t)
    v_W_yh_corrected = v_W_yh / (1 - beta2**t)
    v_b_y_corrected = v_b_y / (1 - beta2**t)
    
    # Update parameters
    W_hx -= learning_rate * m_W_hx_corrected / (np.sqrt(v_W_hx_corrected) + epsilon)
    W_hh -= learning_rate * m_W_hh_corrected / (np.sqrt(v_W_hh_corrected) + epsilon)
    b_h -= learning_rate * m_b_h_corrected / (np.sqrt(v_b_h_corrected) + epsilon)
    W_yh -= learning_rate * m_W_yh_corrected / (np.sqrt(v_W_yh_corrected) + epsilon)
    b_y -= learning_rate * m_b_y_corrected / (np.sqrt(v_b_y_corrected) + epsilon)

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

# Generate character-level training data with embeddings
def generate_char_data_with_embeddings(text, char_to_idx, sequence_length=T):
    """
    Generate character-level training sequences from text using embeddings
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
        
        # Get embeddings for input characters (treat each character as a mini text)
        # For better embeddings, we'll use small context windows around each character
        input_embeddings = []
        for j, char in enumerate(input_seq):
            # Create a small context window around the character
            start_ctx = max(0, i + j - 2)
            end_ctx = min(len(text), i + j + 3)
            context = text[start_ctx:end_ctx]
            
            # Get embedding for the context
            embedding = get_text_embeddings([context])[0]
            input_embeddings.append(embedding)
        
        # Convert target characters to one-hot encoding (for classification)
        Y_seq = []
        for char in target_seq:
            one_hot_target = np.zeros(n_y)
            one_hot_target[char_to_idx[char]] = 1
            Y_seq.append(one_hot_target)
        
        X_data.append(input_embeddings)
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
            
            # Update parameters using Adam optimizer: θ = θ - α * m̂_t / (√v̂_t + ε)
            update_parameters_adam(gradients, learning_rate)
        
        # Track average loss per epoch
        avg_loss = epoch_loss / len(X_data)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    return losses

def generate_text(seed_text, length=100, temperature=1.0):
    """
    Generate text using the trained RNN model with embeddings
    
    Args:
        seed_text: Starting text to prime the generation
        length: Number of characters to generate
        temperature: Controls randomness (lower = more conservative, higher = more random)
    """
    generated = seed_text
    
    # Initialize hidden state
    h = np.zeros((n_h, 1))
    
    # Process seed text to warm up the hidden state
    for i, char in enumerate(seed_text):
        if char in char_to_idx:
            # Create context window around character
            start_ctx = max(0, i - 2)
            end_ctx = min(len(seed_text), i + 3)
            context = seed_text[start_ctx:end_ctx]
            
            # Get embedding for the context
            embedding = get_text_embeddings([context])[0]
            x = embedding.reshape(-1, 1)
            
            # Forward pass
            a = np.dot(W_hx, x) + np.dot(W_hh, h) + b_h
            h = np.tanh(a)
    
    # Generate new characters
    for i in range(length):
        # Get context for the last character
        if len(generated) > 0:
            # Use a longer context window for better generation
            context_length = min(10, len(generated))  # Use last 10 characters as context
            context = generated[-context_length:]
            
            # Get embedding for the context
            embedding = get_text_embeddings([context])[0]
            x = embedding.reshape(-1, 1)
            
            # Forward pass
            a = np.dot(W_hx, x) + np.dot(W_hh, h) + b_h
            h = np.tanh(a)
            z = np.dot(W_yh, h) + b_y
            
            # Apply temperature scaling
            z = z / temperature
            y_hat = softmax(z)
            
            # Sample from the probability distribution
            probabilities = y_hat.flatten()
            
            # Add some randomness for diversity in long text generation
            if i % 50 == 0 and i > 0:  # Every 50 characters, increase diversity slightly
                probabilities = probabilities ** 0.9  # Flatten the distribution slightly
                probabilities = probabilities / np.sum(probabilities)  # Renormalize
            
            char_idx = np.random.choice(len(chars), p=probabilities)
            next_char = idx_to_char[char_idx]
            
            generated += next_char
        else:
            # If no generated text yet, add a space
            generated += ' '
    
    return generated

def generate_long_text_with_breaks(seed_text, length=1000, temperature=1.0, break_interval=200):
    """
    Generate very long text by breaking it into chunks to maintain coherence
    
    Args:
        seed_text: Starting text to prime the generation
        length: Total number of characters to generate
        temperature: Controls randomness
        break_interval: Generate text in chunks of this size
    """
    generated = seed_text
    remaining_length = length
    
    while remaining_length > 0:
        chunk_length = min(break_interval, remaining_length)
        
        # Generate a chunk
        chunk = generate_text(generated[-50:], length=chunk_length, temperature=temperature)
        
        # Remove the seed part and add only the new text
        new_text = chunk[len(generated[-50:]):]
        generated += new_text
        
        remaining_length -= len(new_text)
        
        # Print progress for very long generations
        if length > 500:
            print(f"Generated {len(generated) - len(seed_text)}/{length} characters...")
    
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
    X_train, Y_train = generate_char_data_with_embeddings(train_text, char_to_idx)
    X_test, Y_test = generate_char_data_with_embeddings(test_text, char_to_idx)
    
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
    
    # Generate sample text with different lengths and temperatures
    print("\nGenerating sample text...")
    seed_texts = ["The ", "When ", "From ", "To be ", "Shall I "]
    lengths = [100, 300, 500]  # Different text lengths
    temperatures = [0.5, 0.8, 1.2]  # Different creativity levels
    
    # Generate short samples with different seeds
    for seed in seed_texts:
        print(f"\nSeed: '{seed}' (Length: 150, Temperature: 0.8)")
        generated = generate_text(seed, length=150, temperature=0.8)
        print(f"Generated: {generated}")
        print("-" * 80)
    
    # Generate longer texts with different parameters
    print("\n" + "="*100)
    print("LONGER TEXT GENERATION EXAMPLES")
    print("="*100)
    
    for length in lengths:
        for temp in temperatures:
            seed = "The "
            print(f"\nGenerating {length} characters with temperature {temp}:")
            print(f"Seed: '{seed}'")
            generated = generate_text(seed, length=length, temperature=temp)
            print(f"Generated ({len(generated)} chars): {generated}")
            print("-" * 120)
    
    # Generate very long text (1000+ characters)
    print("\n" + "="*100)
    print("VERY LONG TEXT GENERATION (1000+ characters)")
    print("="*100)
    
    long_seeds = ["Shall I compare thee ", "When in eternal ", "The fair "]
    for seed in long_seeds:
        print(f"\nSeed: '{seed}' (Target: 1000 characters, Temperature: 0.9)")
        long_generated = generate_text(seed, length=1000, temperature=0.9)
        print(f"Generated ({len(long_generated)} chars):")
        print(long_generated)
        print("-" * 150)
    
    # Show a prediction example
    if len(X_test) > 0:
        sample_X = X_test[0]
        sample_Y = Y_test[0]
        
        predictions = predict(sample_X)
        true_labels = [np.argmax(sample_Y[t]) for t in range(len(sample_Y))]
        
        print("\nSample Character Prediction Example:")
        # For embeddings, we can't easily convert back to characters from the input
        # So we'll show the sequence index and the predictions
        predicted_chars = [idx_to_char[predictions[t]] for t in range(len(predictions))]
        true_chars = [idx_to_char[true_labels[t]] for t in range(len(true_labels))]
        
        print(f"Predicted next: {''.join(predicted_chars)}")
        print(f"True next:      {''.join(true_chars)}")