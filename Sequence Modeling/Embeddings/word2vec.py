import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

class Word2Vec:
    def __init__(self, sentences, embedding_dim=10, window_size=2, learning_rate=0.01, 
                 epochs=100, neg_samples=5):
        """
        Initialize Word2Vec model using Skip-gram with Negative Sampling
        
        Parameters:
        - sentences: List of tokenized sentences (list of lists of strings)
        - embedding_dim: Dimension of word embeddings
        - window_size: Context window size
        - learning_rate: Learning rate for gradient descent
        - epochs: Number of training epochs
        - neg_samples: Number of negative samples per positive sample
        """
        self.sentences = sentences
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.neg_samples = neg_samples
        
        # Create vocabulary and mappings
        self._build_vocabulary()  
        
        # Initialize embeddings
        # Input embeddings (target words)
        self.W = np.random.uniform(-0.5 / self.embedding_dim, 0.5 / self.embedding_dim, 
                                  (self.vocab_size, self.embedding_dim))
        # Output embeddings (context words)
        self.W_prime = np.random.uniform(-0.5 / self.embedding_dim, 0.5 / self.embedding_dim, 
                                       (self.vocab_size, self.embedding_dim))
        
        # Build sampling tables for negative sampling
        self._build_sampling_table()
    
    def _build_vocabulary(self):
        """Build vocabulary and word-to-index mappings"""
        # Flatten sentences
        all_words = [word for sentence in self.sentences for word in sentence]
        
        # Get unique words
        unique_words = list(set(all_words))
        self.vocab_size = len(unique_words)
        
        # Create word-to-index and index-to-word mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        self.idx_to_word = {idx: word for idx, word in enumerate(unique_words)}
        
        # Count word frequencies
        self.word_counts = {}
        for word in all_words:
            if word in self.word_counts:
                self.word_counts[word] += 1
            else:
                self.word_counts[word] = 1
        
        # Calculate total number of words
        self.total_words = len(all_words)
    
    def _build_sampling_table(self):
        """
        Build the sampling table for negative sampling
        Uses the unigram distribution raised to the 3/4 power
        """
        # Calculate sampling weights for each word
        sampling_weights = np.zeros(self.vocab_size)
        for word, idx in self.word_to_idx.items():
            sampling_weights[idx] = self.word_counts[word] ** 0.75
        
        # Normalize to create a probability distribution
        self.sampling_weights = sampling_weights / np.sum(sampling_weights)
    
    def _sigmoid(self, x):
        """Compute sigmoid function"""
        return 1 / (1 + np.exp(-x))
    
    def _get_context_pairs(self):
        """Generate all target-context word pairs from sentences"""
        pairs = []
        
        for sentence in self.sentences:
            sentence_indices = [self.word_to_idx[word] for word in sentence]
            
            # For each position in the sentence
            for pos, target_idx in enumerate(sentence_indices):
                # Define context window boundaries
                start = max(0, pos - self.window_size)
                end = min(len(sentence_indices), pos + self.window_size + 1)
                
                # For each word in the context window
                for context_pos in range(start, end):
                    # Skip the target word itself
                    if context_pos != pos:
                        context_idx = sentence_indices[context_pos]
                        pairs.append((target_idx, context_idx))
        
        return pairs
    
    def _sample_negative(self, positive_context_idx, n_samples):
        """Sample negative context words from sampling distribution"""
        # Sample words based on sampling weights
        negative_samples = []
        while len(negative_samples) < n_samples:
            idx = np.random.choice(self.vocab_size, p=self.sampling_weights)
            # Ensure we don't sample the positive context word
            if idx != positive_context_idx:
                negative_samples.append(idx)
        
        return negative_samples
    
    def train(self):
        """Train the Word2Vec model"""
        print("Generating training pairs...")
        pairs = self._get_context_pairs()
        n_pairs = len(pairs)
        
        print(f"Training on {n_pairs} word pairs...")
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            # Shuffle pairs for each epoch
            random.shuffle(pairs)
            
            for i, (target_idx, context_idx) in enumerate(pairs):
                # Forward pass
                # Get target embedding
                target_embed = self.W[target_idx]
                
                # Positive pair
                dot_product = np.dot(target_embed, self.W_prime[context_idx])
                sigmoid_pos = self._sigmoid(dot_product)
                loss_pos = -np.log(sigmoid_pos)
                
                # Negative samples
                neg_indices = self._sample_negative(context_idx, self.neg_samples)
                loss_neg = 0
                
                # Accumulated gradients
                # Initialize gradient for target embedding
                grad_target = np.zeros(self.embedding_dim)
                
                # Gradient for positive context
                context_grad = (sigmoid_pos - 1) * target_embed
                grad_target += (sigmoid_pos - 1) * self.W_prime[context_idx]
                
                # Update positive context embedding
                self.W_prime[context_idx] -= self.learning_rate * context_grad
                
                # Process negative samples
                for neg_idx in neg_indices:
                    # Compute loss and gradients for negative sample
                    dot_product_neg = np.dot(target_embed, self.W_prime[neg_idx])
                    sigmoid_neg = self._sigmoid(dot_product_neg)
                    loss_neg -= np.log(1 - sigmoid_neg)
                    
                    # Gradient for negative context
                    neg_context_grad = sigmoid_neg * target_embed
                    # Add contribution to target gradient
                    grad_target += sigmoid_neg * self.W_prime[neg_idx]
                    
                    # Update negative context embedding
                    self.W_prime[neg_idx] -= self.learning_rate * neg_context_grad
                
                # Update target embedding
                self.W[target_idx] -= self.learning_rate * grad_target
                
                # Track loss
                total_loss += loss_pos + loss_neg
            
            # Decay learning rate
            self.learning_rate = self.learning_rate * 0.99
            
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                avg_loss = total_loss / n_pairs
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def get_word_vector(self, word):
        """Get embedding vector for a word"""
        if word in self.word_to_idx:
            return self.W[self.word_to_idx[word]]
        else:
            return None
    
    def find_similar(self, word, n=5):
        """Find n most similar words to the given word"""
        if word not in self.word_to_idx:
            print(f"Word '{word}' not in vocabulary.")
            return []
        
        word_idx = self.word_to_idx[word]
        word_vec = self.W[word_idx]
        
        # Normalize the query vector
        word_vec = word_vec / np.linalg.norm(word_vec)
        
        # Compute cosine similarities
        similarities = {}
        for idx, vec in enumerate(self.W):
            if idx != word_idx:
                # Normalize the word vector
                vec_norm = vec / np.linalg.norm(vec)
                # Calculate cosine similarity
                similarity = np.dot(word_vec, vec_norm)
                similarities[self.idx_to_word[idx]] = similarity
        
        # Sort by similarity and return top n
        sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:n]
    
    def visualize_embeddings(self, words=None, n=100):
        """
        Visualize word embeddings using t-SNE
        
        Parameters:
        - words: List of specific words to visualize 
                (if None, visualize top n frequent words)
        - n: Number of most frequent words to visualize
        """
        if words is None:
            # Get top n frequent words
            word_freq = [(word, count) for word, count in self.word_counts.items()]
            word_freq.sort(key=lambda x: x[1], reverse=True)
            words = [word for word, _ in word_freq[:n]]
        
        # Get embeddings for the words
        word_indices = [self.word_to_idx[word] for word in words if word in self.word_to_idx]
        
        # Check if we have enough words to visualize
        if len(word_indices) < 2:
            print("Not enough words to visualize. Need at least 2 words.")
            return
            
        embeddings = self.W[word_indices]
        
        # Apply t-SNE for dimensionality reduction
        # Adjust perplexity to be less than n_samples
        perplexity = min(30, len(word_indices) - 1)  # Default is 30, but must be < n_samples
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 10))
        for i, word in enumerate([self.idx_to_word[idx] for idx in word_indices]):
            x, y = embeddings_2d[i, :]
            plt.scatter(x, y, marker='o')
            plt.annotate(word, (x, y), fontsize=9)
        
        plt.title('t-SNE visualization of word embeddings')
        plt.grid(True)
        plt.show()

    def analogy(self, word1, word2, word3, n=5):
        """
        Solve word analogies like "king - man + woman = ?"
        Example: analogy("king", "man", "woman") should return "queen"
        """
        if word1 not in self.word_to_idx or \
           word2 not in self.word_to_idx or \
           word3 not in self.word_to_idx:
            print(f"One or more words not in vocabulary")
            return []
        
        # Get word vectors
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        vec3 = self.get_word_vector(word3)
        
        # Calculate the analogy vector
        analogy_vec = vec1 - vec2 + vec3
        
        # Normalize the analogy vector
        analogy_vec = analogy_vec / np.linalg.norm(analogy_vec)
        
        # Find closest words to the analogy vector
        similarities = {}
        for idx, vec in enumerate(self.W):
            # Skip the input words
            if self.idx_to_word[idx] not in [word1, word2, word3]:
                # Normalize the word vector
                vec_norm = vec / np.linalg.norm(vec)
                # Calculate cosine similarity
                similarity = np.dot(analogy_vec, vec_norm)
                similarities[self.idx_to_word[idx]] = similarity
        
        # Sort and return top n
        sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:n]
    
# Example usage
if __name__ == "__main__":
    # Simple example sentences
    sentences = [
        ["the", "cat", "sits", "on", "the", "mat"],
        ["the", "dog", "runs", "in", "the", "park"],
        ["cats", "and", "dogs", "are", "animals"],
        ["paris", "is", "the", "capital", "of", "france"],
        ["berlin", "is", "the", "capital", "of", "germany"],
        ["rome", "is", "the", "capital", "of", "italy"],
        ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        ["man", "is", "to", "king", "as", "woman", "is", "to", "queen"],
        ["apple", "is", "a", "fruit", "and", "carrot", "is", "a", "vegetable"],
        ["computers", "can", "run", "programs", "and", "process", "data"]
    ]
    
    # Create and train the model
    try:
        model = Word2Vec(sentences, 
                         embedding_dim=50,  # Higher dimension for better results
                         window_size=2, 
                         learning_rate=0.05, 
                         epochs=200, 
                         neg_samples=3)
        model.train()
        
        # Find similar words
        print("\nWords similar to 'cat':")
        similar_words = model.find_similar('cat', n=3)
        for word, similarity in similar_words:
            print(f"{word}: {similarity:.4f}")
        
        # Test analogy
        print("\nAnalogy test: 'king - man + woman = ?'")
        analogies = model.analogy('king', 'man', 'woman', n=3)
        for word, similarity in analogies:
            print(f"{word}: {similarity:.4f}")
        
        # Visualize embeddings - use a smaller number for a small dataset
        try:
            # Get actual vocab size to determine proper visualization count
            viz_count = min(30, len(model.word_to_idx))
            model.visualize_embeddings(n=viz_count)
        except Exception as e:
            print(f"Visualization error: {e}")
            print("Skipping visualization...")
        
        # Demonstrate a complete example for the pair ("cat", "sits")
        print("\nDetailed example for pair ('cat', 'sits'):")
        # Get indices and vectors
        cat_idx = model.word_to_idx['cat']
        sits_idx = model.word_to_idx['sits']
        
        # Get embeddings
        cat_embed = model.W[cat_idx]
        sits_embed = model.W_prime[sits_idx]
        
        # Compute dot product and sigmoid
        dot_product = np.dot(cat_embed, sits_embed)
        sigmoid_value = model._sigmoid(dot_product)
        
        print(f"Target word: 'cat', Context word: 'sits'")
        print(f"cat embedding (first 5 dims): {cat_embed[:5]}")
        print(f"sits embedding (first 5 dims): {sits_embed[:5]}")
        print(f"Dot product: {dot_product:.4f}")
        print(f"Sigmoid value: {sigmoid_value:.4f}")
        print(f"Probability of 'sits' given 'cat': {sigmoid_value:.4f}")
        
        # Generate negative samples
        neg_samples = model._sample_negative(sits_idx, 3)
        neg_words = [model.idx_to_word[idx] for idx in neg_samples]
        print(f"Negative samples: {neg_words}")
        
        # Compute loss
        loss_pos = -np.log(sigmoid_value)
        print(f"Loss for positive sample: {loss_pos:.4f}")
        
    except Exception as e:
        print(f"Error running Word2Vec example: {e}")