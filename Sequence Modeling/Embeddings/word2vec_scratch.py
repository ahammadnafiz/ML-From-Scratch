text = """Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks."""

import re

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


def tokenize(text):
    pattern = re.compile(r"[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*")
    return pattern.findall(text.lower())


tokens = tokenize(text)
print(f"Number of tokens: {len(tokens)}")
print(f"Unique tokens: {len(set(tokens))}")


def mapping(tokens):
    word_to_id = {}
    id_to_word = {}
    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token
    return word_to_id, id_to_word


word_to_id, id_to_word = mapping(tokens)


def one_hot_encode(id, vocab_size):
    res = np.zeros(vocab_size)
    res[id] = 1
    return res


def generate_training_data(tokens, word_to_id, window):
    X = []
    y = []
    n_tokens = len(tokens)

    for i in range(n_tokens):
        # Context words before center word
        for j in range(max(0, i - window), i):
            X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
            y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))

        # Context words after center word
        for j in range(i + 1, min(n_tokens, i + window + 1)):
            X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
            y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))

    return np.array(X), np.array(y)


X, y = generate_training_data(tokens, word_to_id, 2)
print(f"Training data shape: X={X.shape}, y={y.shape}")


def init_network(vocab_size, n_embedding):
    # Xavier/Glorot initialization for better convergence
    w1_bound = np.sqrt(6.0 / (vocab_size + n_embedding))
    w2_bound = np.sqrt(6.0 / (n_embedding + vocab_size))

    model = {
        "w1": np.random.uniform(-w1_bound, w1_bound, (vocab_size, n_embedding)),
        "w2": np.random.uniform(-w2_bound, w2_bound, (n_embedding, vocab_size)),
    }
    return model


model = init_network(len(word_to_id), 50)  # Increased embedding size


def softmax(x):
    # Numerical stability improvements
    x_max = np.max(x, axis=1, keepdims=True)
    x_shifted = x - x_max
    exp_x = np.exp(np.clip(x_shifted, -500, 500))  # Prevent overflow
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def forward(model, X):
    h = X @ model["w1"]  # Hidden layer (embeddings)
    u = h @ model["w2"]  # Output layer scores
    y_pred = softmax(u)  # Probabilities
    return h, u, y_pred


def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))


def backward_pass(model, X, y, learning_rate):
    batch_size = X.shape[0]

    # Forward pass
    h, u, y_pred = forward(model, X)

    # Backward pass
    # Output layer gradients
    dL_du = (y_pred - y) / batch_size
    dL_dW2 = h.T @ dL_du

    # Hidden layer gradients
    dL_dh = dL_du @ model["w2"].T
    dL_dW1 = X.T @ dL_dh

    # Update weights with gradient descent
    model["w2"] -= learning_rate * dL_dW2
    model["w1"] -= learning_rate * dL_dW1

    # Calculate loss
    loss = cross_entropy_loss(y_pred, y)
    return loss


# Training with mini-batches and learning rate scheduling
def train_model(model, X, y, epochs=200, initial_lr=0.1, batch_size=32):
    history = []
    n_samples = X.shape[0]

    print("Training Word2Vec model...")
    print(f"Vocabulary size: {len(word_to_id)}")
    print(f"Embedding dimension: {model['w1'].shape[1]}")
    print(f"Training samples: {n_samples}")
    print("-" * 50)

    for epoch in range(epochs):
        # Learning rate decay
        lr = initial_lr * (0.99**epoch)

        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0
        n_batches = 0

        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X_shuffled[i:end_idx]
            y_batch = y_shuffled[i:end_idx]

            loss = backward_pass(model, X_batch, y_batch, lr)
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss: {avg_loss:.4f}, LR: {lr:.4f}")

    return history


# Train the model
history = train_model(model, X, y, epochs=1000, initial_lr=0.5, batch_size=16)

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history, color="skyblue", linewidth=2)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history[-50:], color="coral", linewidth=2)
plt.title("Loss (Last 50 Epochs)")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Extract word embeddings
def get_word_embeddings(model, word_to_id):
    embeddings = {}
    for word, idx in word_to_id.items():
        embeddings[word] = model["w1"][idx]
    return embeddings


embeddings = get_word_embeddings(model, word_to_id)


# Cosine similarity function
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)


def find_most_similar(target_word, embeddings, top_k=5):
    if target_word not in embeddings:
        return []

    target_embedding = embeddings[target_word]
    similarities = []

    for word, embedding in embeddings.items():
        if word != target_word:
            sim = cosine_similarity(target_embedding, embedding)
            similarities.append((word, sim))

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]


# Test similarity for key words
print("\n" + "=" * 60)
print("WORD SIMILARITY ANALYSIS")
print("=" * 60)

test_words = ["machine", "learning", "algorithms", "data", "model"]
for word in test_words:
    if word in embeddings:
        similar_words = find_most_similar(word, embeddings, top_k=3)
        print(f"\n'{word}' is most similar to:")
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.3f}")

import matplotlib.pyplot as plt
# Visualize embeddings using PCA
from sklearn.decomposition import PCA


def visualize_embeddings(embeddings, words_to_plot=None):
    if words_to_plot is None:
        words_to_plot = list(embeddings.keys())[:15]  # Plot first 15 words

    # Get embeddings for selected words
    word_vectors = []
    labels = []

    for word in words_to_plot:
        if word in embeddings:
            word_vectors.append(embeddings[word])
            labels.append(word)

    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(word_vectors)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)

    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(
            label,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    plt.title("Word Embeddings Visualization (PCA)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Visualize embeddings
try:
    visualize_embeddings(embeddings)
except ImportError:
    print("\nSkipping visualization (sklearn not available)")
    print("Install scikit-learn to see embedding plots: pip install scikit-learn")

print(f"\nFinal training loss: {history[-1]:.4f}")
print(f"Total improvement: {history[0] - history[-1]:.4f}")
