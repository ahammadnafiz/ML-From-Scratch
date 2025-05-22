import numpy as np

# --- Data preparation ---
sentence = ["I", "love", "dogs", "hate", "dogs"]
vocab = list(set(sentence))
word_to_index = {w: i for i, w in enumerate(vocab)}
index_to_word = {i: w for w, i in word_to_index.items()}
V = len(vocab)
N = 2  # Embedding size

# Generate training pairs (center, context) with window size = 1
def generate_pairs(words, window_size=1):
    pairs = []
    for i, center in enumerate(words):
        for j in range(i - window_size, i + window_size + 1):
            if j != i and 0 <= j < len(words):
                pairs.append((center, words[j]))
    return pairs

training_pairs = generate_pairs(sentence)

# --- Model Initialization ---
np.random.seed(42)
W = np.random.randn(V, N) * 0.1      # Embedding matrix
W_out = np.random.randn(N, V) * 0.1  # Output matrix

def one_hot(word):
    vec = np.zeros((V, 1))
    vec[word_to_index[word]] = 1
    return vec

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

# --- Training Loop ---
learning_rate = 0.1
epochs = 100

for epoch in range(epochs):
    total_loss = 0
    for center_word, context_word in training_pairs:
        x = one_hot(center_word)
        y_true = one_hot(context_word)

        # FORWARD
        h = W.T @ x                    # hidden layer (N x 1)
        u = W_out.T @ h                # score (V x 1)
        y_hat = softmax(u)            # softmax output

        # LOSS
        loss = -np.sum(y_true * np.log(y_hat + 1e-9))  # cross-entropy
        total_loss += loss

        # BACKWARD
        du = y_hat - y_true                    # (V x 1)
        dW_out = h @ du.T                      # (N x V)
        dh = W_out @ du                        # (N x 1)
        dW = np.zeros_like(W)
        center_idx = word_to_index[center_word]
        dW[center_idx] = dh.flatten()          # only update center word row

        # UPDATE
        W -= learning_rate * dW
        W_out -= learning_rate * dW_out

    if epoch % 10 == 0:
        print(f"Epoch {epoch} — Loss: {total_loss:.4f}")

# --- Show learned embeddings ---
print("\nLearned Embeddings:")
for word, idx in word_to_index.items():
    print(f"{word}: {W[idx]}")