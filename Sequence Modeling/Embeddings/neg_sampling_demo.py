import numpy as np
import random

# 1. Toy Corpus - simple example text
corpus = "the quick brown fox jumps over the lazy dog".split()

# 2. Build vocabulary and mappings between words and indices
vocab = list(set(corpus))  # Unique words
word2idx = {word: idx for idx, word in enumerate(vocab)}  # Word → index mapping
idx2word = {idx: word for word, idx in word2idx.items()}  # Index → word mapping
vocab_size = len(vocab)  # Number of unique words in vocabulary

# 3. Generate (center, context) word pairs within a context window
def generate_training_data(corpus, window_size=2):
    pairs = []
    for i, center_word in enumerate(corpus):
        # For each position in corpus, look at surrounding words within window size
        for j in range(max(i - window_size, 0), min(i + window_size + 1, len(corpus))):
            if i != j:  # Skip the center word itself
                # Store word indices instead of words for efficiency
                pairs.append((word2idx[corpus[i]], word2idx[corpus[j]]))
    return pairs

training_pairs = generate_training_data(corpus)

# 4. Create negative sampling distribution based on word frequencies
# Calculate frequency of each word in vocabulary
word_freqs = np.array([corpus.count(word) for word in vocab])  
word_freqs = word_freqs / word_freqs.sum()  # Normalize to probabilities
# Apply power of 0.75 to smooth the distribution (emphasize less frequent words)
neg_sample_probs = word_freqs ** 0.75  
neg_sample_probs = neg_sample_probs / neg_sample_probs.sum()  # Re-normalize

def get_negative_samples(k, exclude_idx):
    """
    Sample k negative words, excluding a specific index.
    
    Args:
        k: Number of negative samples
        exclude_idx: Index to exclude (typically the positive context word)
    
    Returns:
        List of k negative sample indices
    """
    samples = []
    while len(samples) < k:
        # Sample from vocabulary according to neg_sample_probs distribution
        sample = np.random.choice(vocab_size, p=neg_sample_probs)
        if sample != exclude_idx:  # Avoid sampling the positive example
            samples.append(sample)
    return samples

# 5. Initialize word embeddings with small random values
embedding_dim = 10  # Dimension of embedding vectors
# U: Output vectors (when words appear as context)
U = np.random.randn(vocab_size, embedding_dim) * 0.01  
# V: Input vectors (when words appear as center)
V = np.random.randn(vocab_size, embedding_dim) * 0.01  

# 6. Sigmoid function with clipping to prevent numerical overflow
def sigmoid(x):
    x = np.clip(x, -500, 500)  # Prevent extreme values that can cause overflow
    return 1 / (1 + np.exp(-x))

# 7. Training loop
def train_sgns(epochs=100, learning_rate=0.05, k=5):
    """
    Train Skip-gram with Negative Sampling model
    
    Args:
        epochs: Number of training iterations over all pairs
        learning_rate: Step size for gradient descent
        k: Number of negative samples per positive sample
    """
    global U, V
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(training_pairs)  # Randomize training order
        
        for center_idx, context_idx in training_pairs:
            # Get embeddings for current word pair
            v_c = V[center_idx]  # Input vector for center word: shape (embedding_dim,)
            u_o = U[context_idx]  # Output vector for context word: shape (embedding_dim,)

            # FORWARD PASS: POSITIVE SAMPLE
            score_pos = np.dot(u_o, v_c)  # Dot product: shape scalar
            loss_pos = -np.log(sigmoid(score_pos))  # -log(σ(uᵀv)): shape scalar
            
            # BACKWARD PASS: POSITIVE SAMPLE
            # Compute gradients using chain rule
            # Gradient for context word: ∂L/∂u_o = (σ(uᵀv) - 1)v_c
            grad_u_o = (sigmoid(score_pos) - 1) * v_c  # shape (embedding_dim,)
            # Initial gradient for center word: ∂L/∂v_c = (σ(uᵀv) - 1)u_o
            grad_v_c = (sigmoid(score_pos) - 1) * u_o  # shape (embedding_dim,)

            # Update output vector for positive context word
            U[context_idx] -= learning_rate * grad_u_o

            # FORWARD & BACKWARD PASS: NEGATIVE SAMPLES
            neg_indices = get_negative_samples(k, context_idx)  # Get k negative samples
            for neg_idx in neg_indices:
                u_k = U[neg_idx]  # Output vector for negative word: shape (embedding_dim,)
                
                # FORWARD PASS: compute negative sample loss
                score_neg = np.dot(u_k, v_c)  # Dot product: shape scalar
                loss_neg = -np.log(sigmoid(-score_neg))  # -log(σ(-uᵏᵀv)): shape scalar
                
                # BACKWARD PASS: compute gradients for negative sample
                # Gradient for negative word: ∂L/∂u_k = (1-σ(-uᵏᵀv))v_c
                grad_u_k = (1 - sigmoid(-score_neg)) * v_c  # shape (embedding_dim,)
                # Accumulate gradient for center word: ∂L/∂v_c += (1-σ(-uᵏᵀv))u_k
                grad_v_c += (1 - sigmoid(-score_neg)) * u_k  # shape (embedding_dim,)

                # Update output vector for negative word
                U[neg_idx] -= learning_rate * grad_u_k
                
                # Accumulate loss from negative sample
                loss_pos += loss_neg

            # Update center word embedding using accumulated gradient
            V[center_idx] -= learning_rate * grad_v_c
            total_loss += loss_pos  # Add current pair's loss to epoch total

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {total_loss:.4f}")

# 8. Train the model
train_sgns()

# 9. Check learned embeddings
def get_embedding(word):
    """Retrieve the embedding vector for a given word"""
    return V[word2idx[word]]

print("\nEmbedding for 'fox':", get_embedding("fox"))