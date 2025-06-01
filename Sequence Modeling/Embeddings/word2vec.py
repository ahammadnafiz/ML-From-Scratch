import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def tokenize(text):
    """Convert text to lowercase tokens"""
    return re.findall(r'\b[a-z]+\b', text.lower())

def build_vocabulary(tokens, min_count=1):
    """Create word-to-index and index-to-word mappings"""
    word_counts = defaultdict(int)
    
    # Count word frequencies
    for token in tokens:
        word_counts[token] += 1
    
    # Filter by minimum count and create mappings
    vocab = {}
    idx_to_word = {}
    
    # Sort by frequency to ensure consistent ordering
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    idx = 0
    for word, count in sorted_words:
        if count >= min_count:
            vocab[word] = idx
            idx_to_word[idx] = word
            idx += 1
    
    return vocab, idx_to_word

def generate_training_pairs(tokens, vocab, window_size=2):
    """Create (center_word, context_word) pairs for training"""
    pairs = []
    
    for i, center_word in enumerate(tokens):
        if center_word not in vocab:
            continue
        
        # Define context window bounds
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        # Collect context words
        for j in range(start, end):
            if i != j and tokens[j] in vocab:
                center_idx = vocab[center_word]
                context_idx = vocab[tokens[j]]
                pairs.append((center_idx, context_idx))
    
    return pairs

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim=100):
        """Initialize embedding matrices"""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # W₁: Input embeddings (V × N) - Xavier initialization
        self.W1 = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
        
        # W₂: Output embeddings (N × V) - Xavier initialization  
        self.W2 = np.random.normal(0, 0.1, (embedding_dim, vocab_size))
    
    def sigmoid(self, x):
        """Numerically stable sigmoid function"""
        x = np.clip(x, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-x))
    
    def forward_pass(self, center_idx, context_idx):
        """Compute forward pass for single training pair"""
        
        # Step 1: Embedding lookup (Layer 1)
        # L₁ = XW₁ where X is one-hot
        center_embedding = self.W1[center_idx, :]  # Shape: (embedding_dim,)
        
        # Step 2: Compute scores (Layer 2)  
        # L₂ = L₁W₂
        scores = np.dot(center_embedding, self.W2)  # Shape: (vocab_size,)
        
        # Step 3: Softmax probabilities (Layer 3)
        # Numerically stable softmax
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / np.sum(exp_scores)
        
        return center_embedding, scores, probabilities
    
    def backward_pass(self, center_idx, context_idx, center_embedding, 
                      probabilities, learning_rate=0.01):
        """Compute gradients and update weights"""
        
        # Step 1: Compute output gradient
        # ∂L/∂scores = probabilities - one_hot_target
        grad_scores = probabilities.copy()
        grad_scores[context_idx] -= 1  # Subtract 1 for correct word
        
        # Step 2: Compute W₂ gradient
        # ∂L/∂W₂ = center_embedding^T ⊗ grad_scores
        grad_W2 = np.outer(center_embedding, grad_scores)
        
        # Step 3: Compute center embedding gradient  
        # ∂L/∂center_embedding = W₂ @ grad_scores
        grad_center_embedding = np.dot(self.W2, grad_scores)
        
        # Step 4: Update weights
        self.W2 -= learning_rate * grad_W2.T  # Note the transpose
        self.W1[center_idx, :] -= learning_rate * grad_center_embedding
        
        # Step 5: Compute loss for monitoring
        loss = -np.log(probabilities[context_idx] + 1e-10)  # Add small value for stability
        return loss
    
    def train_with_negative_sampling(self, center_idx, context_idx, 
                                    negative_samples, learning_rate=0.01):
        """Train using negative sampling (much faster)"""
        
        center_embedding = self.W1[center_idx, :]
        
        # Initialize gradients
        grad_center = np.zeros_like(center_embedding)
        total_loss = 0
        
        # Positive sample: actual context word
        context_vec = self.W2[:, context_idx]
        score = np.dot(center_embedding, context_vec)
        # Clip score to prevent overflow
        score = np.clip(score, -10, 10)
        sigmoid_score = self.sigmoid(score)
        
        # Loss and gradients for positive sample
        loss = -np.log(sigmoid_score + 1e-10)
        grad = (sigmoid_score - 1)  # ∂L/∂score for positive sample
        
        grad_center += grad * context_vec
        self.W2[:, context_idx] -= learning_rate * grad * center_embedding
        total_loss += loss
        
        # Negative samples: random words
        for neg_idx in negative_samples:
            if neg_idx == center_idx or neg_idx == context_idx:
                continue
                
            neg_vec = self.W2[:, neg_idx]
            score = np.dot(center_embedding, neg_vec)
            # Clip score to prevent overflow
            score = np.clip(score, -10, 10)
            sigmoid_score = self.sigmoid(score)
            
            # Loss and gradients for negative sample
            loss = -np.log(1 - sigmoid_score + 1e-10)
            grad = sigmoid_score  # ∂L/∂score for negative sample
            
            grad_center += grad * neg_vec
            self.W2[:, neg_idx] -= learning_rate * grad * center_embedding
            total_loss += loss
        
        # Update center word embedding with gradient clipping
        grad_center = np.clip(grad_center, -1, 1)
        self.W1[center_idx, :] -= learning_rate * grad_center
        
        return total_loss

def train_word2vec(text, embedding_dim=100, window_size=2, 
                   negative_samples=5, epochs=5, learning_rate=0.01):
    """Complete training pipeline"""
    
    # Data preparation
    tokens = tokenize(text)
    vocab, idx_to_word = build_vocabulary(tokens, min_count=1)  # Reduced min_count for small corpus
    training_pairs = generate_training_pairs(tokens, vocab, window_size)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training pairs: {len(training_pairs)}")
    print(f"Tokens: {len(tokens)}")
    
    if len(vocab) < 3:
        print("Warning: Vocabulary too small for meaningful training")
        return None, None, None
    
    # Initialize model
    model = Word2Vec(len(vocab), embedding_dim)
    
    # Adjust negative samples if vocabulary is small
    actual_negative_samples = min(negative_samples, max(1, len(vocab) - 2))
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(training_pairs)  # Shuffle for better convergence
        
        for center_idx, context_idx in training_pairs:
            # Sample negative examples (ensure we don't exceed vocab size)
            if actual_negative_samples > 0:
                # Create a list excluding center and context words
                available_indices = [i for i in range(len(vocab)) if i != center_idx and i != context_idx]
                if len(available_indices) >= actual_negative_samples:
                    negative_idxs = np.random.choice(
                        available_indices, 
                        size=actual_negative_samples, 
                        replace=False
                    )
                else:
                    negative_idxs = available_indices
            else:
                negative_idxs = []
            
            # Train on this pair
            if len(negative_idxs) > 0:
                loss = model.train_with_negative_sampling(
                    center_idx, context_idx, negative_idxs, learning_rate
                )
                total_loss += loss
        
        if training_pairs:
            avg_loss = total_loss / len(training_pairs)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return model, vocab, idx_to_word

def find_similar_words(model, word, vocab, idx_to_word, top_k=5):
    """Find most similar words using cosine similarity"""
    if word not in vocab:
        return f"Word '{word}' not in vocabulary"
    
    word_idx = vocab[word]
    word_vec = model.W1[word_idx, :]  # Get embedding
    
    similarities = []
    for idx, other_word in idx_to_word.items():
        if idx != word_idx:
            other_vec = model.W1[idx, :]
            
            # Cosine similarity: cos(θ) = (a·b)/(|a||b|)
            norm_word = np.linalg.norm(word_vec)
            norm_other = np.linalg.norm(other_vec)
            
            if norm_word > 0 and norm_other > 0:
                cosine_sim = np.dot(word_vec, other_vec) / (norm_word * norm_other)
            else:
                cosine_sim = 0
                
            similarities.append((cosine_sim, other_word))
    
    # Sort by similarity (descending)
    similarities.sort(reverse=True)
    
    print(f"\nWords most similar to '{word}':")
    for i, (sim, similar_word) in enumerate(similarities[:top_k]):
        print(f"{i+1}. {similar_word} (similarity: {sim:.3f})")

def test_analogy(model, vocab, idx_to_word, a, b, c, top_k=3):
    """Test analogy: a is to b as c is to ?"""
    if not all(word in vocab for word in [a, b, c]):
        return "Some words not in vocabulary"
    
    # Get embeddings
    vec_a = model.W1[vocab[a], :]
    vec_b = model.W1[vocab[b], :]  
    vec_c = model.W1[vocab[c], :]
    
    # Vector arithmetic: king - man + woman ≈ queen
    target_vec = vec_b - vec_a + vec_c
    
    # Find closest word to target vector
    similarities = []
    for word, idx in vocab.items():
        if word not in [a, b, c]:  # Exclude input words
            word_vec = model.W1[idx, :]
            
            norm_target = np.linalg.norm(target_vec)
            norm_word = np.linalg.norm(word_vec)
            
            if norm_target > 0 and norm_word > 0:
                similarity = np.dot(target_vec, word_vec) / (norm_target * norm_word)
            else:
                similarity = 0
                
            similarities.append((similarity, word))
    
    similarities.sort(reverse=True)
    
    print(f"\n'{a}' is to '{b}' as '{c}' is to:")
    for i, (sim, word) in enumerate(similarities[:top_k]):
        print(f"{i+1}. {word} (similarity: {sim:.3f})")

def visualize_embeddings(model, vocab, idx_to_word, words_to_show=20):
    """Visualize embeddings in 2D using PCA"""
    
    # Get embeddings for most frequent words
    embeddings = []
    labels = []
    
    for i, word in enumerate(idx_to_word.values()):
        if i < words_to_show:
            embeddings.append(model.W1[i, :])
            labels.append(word)
    
    embeddings = np.array(embeddings)
    
    # Reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=100)
    
    for i, label in enumerate(labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.title("Word2Vec Embeddings (2D PCA)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.grid(True, alpha=0.3)
    plt.show()

def print_embedding(model, word, vocab):
    """Print the raw embedding vector for a word"""
    if word not in vocab:
        print(f"Word '{word}' not in vocabulary")
        return
    
    word_idx = vocab[word]
    embedding = model.W1[word_idx, :]
    
    print(f"\nEmbedding for '{word}':")
    print(f"Shape: {embedding.shape}")
    print(f"Values: {embedding}")
    print(f"Norm: {np.linalg.norm(embedding):.4f}")

def print_all_embeddings(model, vocab, idx_to_word, max_words=10):
    """Print embeddings for multiple words"""
    print("\n" + "="*60)
    print("WORD EMBEDDINGS")
    print("="*60)
    
    count = 0
    for idx, word in idx_to_word.items():
        if count >= max_words:
            break
        print_embedding(model, word, vocab)
        count += 1

def compare_embeddings(model, vocab, word1, word2):
    """Compare two word embeddings"""
    if word1 not in vocab or word2 not in vocab:
        print("One or both words not in vocabulary")
        return
    
    emb1 = model.W1[vocab[word1], :]
    emb2 = model.W1[vocab[word2], :]
    
    # Calculate similarity
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 > 0 and norm2 > 0:
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
    else:
        similarity = 0
    
    print(f"\nComparing '{word1}' and '{word2}':")
    print(f"'{word1}' embedding: {emb1}")
    print(f"'{word2}' embedding: {emb2}")
    print(f"Cosine similarity: {similarity:.4f}")
    print(f"Euclidean distance: {np.linalg.norm(emb1 - emb2):.4f}")

# Example usage and testing
if __name__ == "__main__":
    # Sample text corpus
    sample_text = """
    The king and the queen ruled the kingdom.  
The prince and the princess were the children of the royal family.  
A man and a woman walked through the village where the farmers lived.  
The capital of France is Paris. The capital of Germany is Berlin.  
Paris is known for fashion, art, and culture. Berlin is famous for history and architecture.  
The dog barked at the cat, while the horse galloped across the field.  
The teacher taught students in the classroom. The professor lectured at the university.  
Apples and oranges are fruits. Carrots and potatoes are vegetables.  
The sun rises in the east and sets in the west.  
Music brings joy to people, while silence brings peace to the mind.  

    """
    
    print("Training Word2Vec model...")
    
    # Train the model with reduced parameters for small vocabulary
    model, vocab, idx_to_word = train_word2vec(
        sample_text,
        embedding_dim=20,  # Smaller embedding for small vocab
        window_size=3, 
        epochs=200,
        learning_rate=0.011,  # Reduced learning rate
        negative_samples=3  # Fewer negative samples for small vocab
    )
    
    print("\n" + "="*50)
    print("TESTING RESULTS")
    print("="*50)
    
    # Test similarity
    test_words = ['king', 'paris', 'teacher', 'dog', 'sun']
    for word in test_words:
        if word in vocab:
            find_similar_words(model, word, vocab, idx_to_word)
    
    # Test analogies
    print("\n" + "="*30)
    print("ANALOGY TESTS")
    print("="*30)
    
    analogies = [
        ('king', 'queen', 'man'),         # man is to woman as king is to queen
        ('paris', 'france', 'berlin'),    # berlin is to germany as paris is to france
        ('teacher', 'classroom', 'professor'),  # professor is to university as teacher is to classroom
        ('apples', 'fruits', 'carrots'),  # carrots are to vegetables as apples are to fruits
        ('dog', 'cat', 'horse')           # horse is to field as dog is to cat
    ]
    
    for a, b, c in analogies:
        if all(word in vocab for word in [a, b, c]):
            test_analogy(model, vocab, idx_to_word, a, b, c)
    
    # Print vocabulary for reference
    print(f"\nVocabulary ({len(vocab)} words):")
    print(list(vocab.keys()))
    
    # Print some raw embeddings
    print_all_embeddings(model, vocab, idx_to_word, max_words=5)
    
    # Compare specific word pairs
    print("\n" + "="*30)
    print("EMBEDDING COMPARISONS")
    print("="*30)
    compare_embeddings(model, vocab, 'cat', 'dog')
    compare_embeddings(model, vocab, 'king', 'queen')
    
    # Visualize embeddings (uncomment to see the plot)
    visualize_embeddings(model, vocab, idx_to_word, words_to_show=15)
