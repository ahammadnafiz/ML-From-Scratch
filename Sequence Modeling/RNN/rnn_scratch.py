import os
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

def load_text_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"Text length: {len(text)}")
    print(f"Vocab size: {len(chars)}")

    return text, chars, char_to_idx, idx_to_char

def char_to_onehot(char, char_to_idx, vocab_size):
    """Convert character to one-hot encoding"""
    onehot = np.zeros((vocab_size, 1))
    onehot[char_to_idx[char]] = 1
    return onehot

def create_sequences(text, char_to_idx, sequence_length):
    """Create input-output sequence pairs for training"""
    X_sequences, Y_sequences = [], []
    
    for i in range(len(text) - sequence_length):
        # Input sequence
        input_chars = text[i:i + sequence_length]
        # Target sequence (shifted by 1)
        target_chars = text[i + 1:i + sequence_length + 1]
        
        X_seq = [char_to_onehot(ch, char_to_idx, len(char_to_idx)) for ch in input_chars]
        Y_seq = [char_to_onehot(ch, char_to_idx, len(char_to_idx)) for ch in target_chars]
        
        X_sequences.append(X_seq)
        Y_sequences.append(Y_seq)
    
    return X_sequences, Y_sequences


script_dir = os.path.dirname(os.path.abspath(__file__))
text_file_path = os.path.join(script_dir, "demo.txt")
text, chars, char_to_idx, idx_to_char = load_text_data(text_file_path)

# Create training sequences
sequence_length = 10  # Increase sequence length for better context
X_data, Y_data = create_sequences(text, char_to_idx, sequence_length)
print(f"Number of training sequences: {len(X_data)}")

# Model parameters
n_x = len(chars)  # vocab size for one-hot encoding
n_h = 128         # Increase hidden layer size
n_y = len(chars)  # output vocab size
T = sequence_length

# Initialize weights with better initialization
W_hx = np.random.randn(n_h, n_x) * 0.01  # Smaller initialization
W_hh = np.random.randn(n_h, n_h) * 0.01  # Smaller initialization  
W_yh = np.random.randn(n_y, n_h) * 0.01  # Smaller initialization

b_h = np.zeros((n_h, 1))
b_y = np.zeros((n_y, 1))

m_W_hx = np.zeros_like(W_hx)
m_W_hh = np.zeros_like(W_hh)
m_b_h = np.zeros_like(b_h)
m_W_yh = np.zeros_like(W_yh)
m_b_y = np.zeros_like(b_y)

v_W_hx = np.zeros_like(W_hx)
v_W_hh = np.zeros_like(W_hh)
v_b_h = np.zeros_like(b_h)
v_W_yh = np.zeros_like(W_yh)
v_b_y = np.zeros_like(b_y)

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
t = 0

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - np.tanh(x) ** 2

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def rnn_forward(X, Y):
    h = {-1: np.zeros((n_h, 1))}
    a, z, y_hat = {}, {}, {}
    loss = 0

    for t in range(len(X)):
        x_t = X[t]  # X[t] is already a one-hot vector of shape (vocab_size, 1)
        a[t] = np.dot(W_hx, x_t) + np.dot(W_hh, h[t - 1]) + b_h
        h[t] = np.tanh(a[t])
        z[t] = np.dot(W_yh, h[t]) + b_y
        y_hat[t] = softmax(z[t])
        loss += -np.sum(Y[t] * np.log(y_hat[t] + epsilon))

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

    for t in reversed(range(len(X))):
        dz = y_hat[t] - Y[t]
        dW_yh += np.dot(dz, h[t].T)
        db_y += dz
        dh = np.dot(W_yh.T, dz) + dh_next
        da = dtanh(a[t]) * dh
        dW_hh += np.dot(da, h[t - 1].T)
        dW_hx += np.dot(da, X[t].T)
        db_h += da
        dh_next = np.dot(W_hh.T, da)

    for dparam in [dW_hx, dW_hh, db_h, dW_yh, db_y]:
        np.clip(dparam, -1, 1, out=dparam)  # Less aggressive clipping

    gradients = {
        "dW_hx": dW_hx,
        "dW_hh": dW_hh,
        "db_h": db_h,
        "dW_yh": dW_yh,
        "db_y": db_y,
    }

    return gradients

def update_parameters_adam(gradients, learning_rate=0.001):
    global W_hx, W_hh, b_h, W_yh, b_y
    global m_W_hx, m_W_hh, m_b_h, m_W_yh, m_b_y
    global v_W_hx, v_W_hh, v_b_h, v_W_yh, v_b_y
    global t

    # Increment time step
    t += 1

    # Update first moment (momentum) for each parameter
    m_W_hx = beta1 * m_W_hx + (1 - beta1) * gradients["dW_hx"]
    m_W_hh = beta1 * m_W_hh + (1 - beta1) * gradients["dW_hh"]
    m_b_h = beta1 * m_b_h + (1 - beta1) * gradients["db_h"]
    m_W_yh = beta1 * m_W_yh + (1 - beta1) * gradients["dW_yh"]
    m_b_y = beta1 * m_b_y + (1 - beta1) * gradients["db_y"]

    # Update second moment (RMSprop) for each parameter
    v_W_hx = beta2 * v_W_hx + (1 - beta2) * np.square(gradients["dW_hx"])
    v_W_hh = beta2 * v_W_hh + (1 - beta2) * np.square(gradients["dW_hh"])
    v_b_h = beta2 * v_b_h + (1 - beta2) * np.square(gradients["db_h"])
    v_W_yh = beta2 * v_W_yh + (1 - beta2) * np.square(gradients["dW_yh"])
    v_b_y = beta2 * v_b_y + (1 - beta2) * np.square(gradients["db_y"])

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
    """Predict next characters given a sequence of one-hot encoded characters"""
    h = {}
    h[-1] = np.zeros((n_h, 1))
    y_hat = {}

    for t in range(len(X)):
        x_t = X[t]  # X[t] is already a one-hot vector
        h[t] = tanh(np.dot(W_hx, x_t) + np.dot(W_hh, h[t - 1]) + b_h)
        z_t = np.dot(W_yh, h[t]) + b_y
        y_hat[t] = softmax(z_t)

    prediction = [np.argmax(y_hat[t]) for t in range(len(X))]
    return prediction

def generate_text(seed_chars, length=100, temperature=1.0):
    """Generate text using the trained model"""
    generated = seed_chars
    h = np.zeros((n_h, 1))
    
    # Process seed characters
    for char in seed_chars:
        if char in char_to_idx:  # Check if character exists in vocabulary
            x = char_to_onehot(char, char_to_idx, n_y)
            h = tanh(np.dot(W_hx, x) + np.dot(W_hh, h) + b_h)
    
    # Generate new characters
    for _ in range(length):
        # Get last character as input
        last_char = generated[-1]
        if last_char not in char_to_idx:
            last_char = ' '  # Fallback to space if character not in vocab
            
        x = char_to_onehot(last_char, char_to_idx, n_y)
        h = tanh(np.dot(W_hx, x) + np.dot(W_hh, h) + b_h)
        z = np.dot(W_yh, h) + b_y
        y_pred = softmax(z / temperature)
        
        # Sample from the probability distribution
        probabilities = y_pred.flatten()
        probabilities = probabilities / np.sum(probabilities)  # Ensure probabilities sum to 1
        char_idx = np.random.choice(range(len(probabilities)), p=probabilities)
        generated += idx_to_char[char_idx]
    
    return generated

def train_model(X_data, Y_data, epochs=1000, learning_rate=0.001, print_every=100):
    """Train the character-level RNN language model"""
    losses = []
    best_loss = float('inf')
    patience = 0
    max_patience = 50
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(X_data))
        
        for i in indices:
            X_seq = X_data[i]
            Y_seq = Y_data[i]
            
            # Forward pass
            loss, cache = rnn_forward(X_seq, Y_seq)
            total_loss += loss
            
            # Backward pass
            gradients = rnn_backward(cache)
            
            # Update parameters with adaptive learning rate
            current_lr = learning_rate * (0.98 ** (epoch // 50))  # Decay learning rate
            update_parameters_adam(gradients, current_lr)
        
        avg_loss = total_loss / len(X_data)
        losses.append(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            
        if patience > max_patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % print_every == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
            
            # Generate sample text
            if epoch > 0:
                sample_text = generate_text("The", length=50, temperature=1.2)
                print(f"Sample: {sample_text}")
                print("-" * 50)
    
    return losses

def plot_loss(losses):
    """Plot training loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def calculate_perplexity(loss):
    """Calculate perplexity from loss"""
    return np.exp(loss)

def save_model_params():
    """Save model parameters"""
    params = {
        'W_hx': W_hx, 'W_hh': W_hh, 'W_yh': W_yh,
        'b_h': b_h, 'b_y': b_y,
        'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char
    }
    return params

# Train the model
print("Starting training...")
losses = train_model(X_data, Y_data, epochs=1000, learning_rate=0.001, print_every=25)

# Plot training progress
plot_loss(losses)

# Generate some sample text
print("\n" + "="*50)
print("GENERATING TEXT SAMPLES")
print("="*50)

seed_phrases = ["The", "From", "But", "And"]
for seed in seed_phrases:
    generated = generate_text(seed, length=100, temperature=0.7)
    print(f"\nSeed: '{seed}'")
    print(f"Generated: {generated}")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETE!")
print("="*50)