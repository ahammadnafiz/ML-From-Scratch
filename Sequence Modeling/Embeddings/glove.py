import numpy as np
from collections import defaultdict
from math import log

# Toy vocab
vocab = ["the", "cat", "sat"]
word2idx = {w: i for i, w in enumerate(vocab)}

# Co-occurrence matrix (from manual counts)
X = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
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