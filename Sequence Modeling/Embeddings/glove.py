import numpy as np
from collections import defaultdict
from math import log

# Toy vocab
vocab = ['I', 'like', 'enjoy', 'deep', 'learning', 'nlp', 'flying']
word2idx = {w: i for i, w in enumerate(vocab)}

# Co-occurrence matrix (from manual counts)
X = np.array([
    [0, 2, 1, 0, 0, 0, 0, 0],
    [2, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]
])

# Hyperparameters
embedding_dim = 2
x_max = 100
alpha = 0.75
lr = 0.05

# Initialize
V = len(vocab)
W = np.random.randn(V, embedding_dim)
W_tilde = np.random.randn(V, embedding_dim)
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

# Training loop
for epoch in range(1000):
    total_loss = 0
    for i in range(V):
        for j in range(V):
            x = X[i][j]
            if x == 0:
                continue
            weight = f(x)
            log_x = log(x)
            dot = np.dot(W[i], W_tilde[j])
            loss = (dot + b[i] + b_tilde[j] - log_x) ** 2
            total_loss += 0.5 * weight * loss

            # Gradients
            grad = weight * (dot + b[i] + b_tilde[j] - log_x)
            W[i] -= lr * grad * W_tilde[j]
            W_tilde[j] -= lr * grad * W[i]
            b[i] -= lr * grad
            b_tilde[j] -= lr * grad
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Final vectors
final_vecs = W + W_tilde
final_embeddings = {word: final_vecs[word2idx[word]] for word in vocab}
print("Final embeddings:")
for word, vec in final_embeddings.items():
    print(f"{word}: {vec}")
    
print(cosine_similarity(final_embeddings['deep'], final_embeddings['I']))