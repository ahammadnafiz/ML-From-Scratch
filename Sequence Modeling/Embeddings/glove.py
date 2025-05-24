import numpy as np
from collections import defaultdict
from math import log

# Toy vocab
vocab = ['I', 'like', 'enjoy', 'deep', 'learning', 'nlp', 'flying']
word2idx = {w: i for i, w in enumerate(vocab)}

# Co-occurrence matrix (fixed to 7x7 to match vocab size)
X = np.array([
    [0, 2, 1, 0, 0, 0, 0],  # I
    [2, 0, 0, 1, 0, 1, 0],  # like
    [1, 0, 0, 0, 0, 0, 1],  # enjoy
    [0, 1, 0, 0, 1, 0, 0],  # deep
    [0, 0, 0, 1, 0, 0, 1],  # learning
    [0, 1, 0, 0, 0, 0, 1],  # nlp
    [0, 0, 1, 0, 1, 1, 0]   # flying
])

# Hyperparameters
embedding_dim = 2
x_max = 100
alpha = 0.75
lr = 0.05

# Initialize
V = len(vocab)
W = np.random.randn(V, embedding_dim) * 0.1  # Smaller initialization
W_tilde = np.random.randn(V, embedding_dim) * 0.1
b = np.zeros(V)
b_tilde = np.zeros(V)

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)

def f(x):
    return (x / x_max) ** alpha if x < x_max else 1

# Training loop with improved gradient updates
for epoch in range(1000):
    total_loss = 0
    
    # Store gradients to avoid order dependency
    W_grad = np.zeros_like(W)
    W_tilde_grad = np.zeros_like(W_tilde)
    b_grad = np.zeros_like(b)
    b_tilde_grad = np.zeros_like(b_tilde)
    
    for i in range(V):
        for j in range(V):
            x = X[i][j]
            if x == 0:
                continue
            
            weight = f(x)
            log_x = log(x)
            dot = np.dot(W[i], W_tilde[j])
            diff = dot + b[i] + b_tilde[j] - log_x
            loss = diff ** 2
            total_loss += 0.5 * weight * loss

            # Accumulate gradients
            grad = weight * diff
            W_grad[i] += grad * W_tilde[j]
            W_tilde_grad[j] += grad * W[i]
            b_grad[i] += grad
            b_tilde_grad[j] += grad
    
    # Apply gradients
    W -= lr * W_grad
    W_tilde -= lr * W_tilde_grad
    b -= lr * b_grad
    b_tilde -= lr * b_tilde_grad
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Final vectors (sum of main and context vectors)
final_vecs = W + W_tilde
final_embeddings = {word: final_vecs[word2idx[word]] for word in vocab}

print("\nFinal embeddings:")
for word, vec in final_embeddings.items():
    print(f"{word}: [{vec[0]:.4f}, {vec[1]:.4f}]")

print(f"\nCosine similarity between 'deep' and 'I': {cosine_similarity(final_embeddings['deep'], final_embeddings['I']):.4f}")

# Let's also check some other interesting pairs
print(f"Cosine similarity between 'deep' and 'learning': {cosine_similarity(final_embeddings['deep'], final_embeddings['learning']):.4f}")
print(f"Cosine similarity between 'like' and 'enjoy': {cosine_similarity(final_embeddings['like'], final_embeddings['enjoy']):.4f}")
print(f"Cosine similarity between 'nlp' and 'flying': {cosine_similarity(final_embeddings['nlp'], final_embeddings['flying']):.4f}")

# Visualize the co-occurrence matrix for reference
print("\nCo-occurrence matrix:")
print("     ", end="")
for word in vocab:
    print(f"{word:>8}", end="")
print()
for i, word in enumerate(vocab):
    print(f"{word:>4} ", end="")
    for j in range(V):
        print(f"{X[i][j]:>8}", end="")
    print()