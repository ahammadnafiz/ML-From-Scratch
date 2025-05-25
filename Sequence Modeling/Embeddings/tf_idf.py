import re  # For regular expressions to clean text
import numpy as np  # For mathematical operations
from collections import Counter  # Efficient counting of elements

class TFIDFVectorizer:
    def __init__(self):
        # Dictionary mapping words to their indices in the vector space
        self.vocabulary = {}
        # List to store IDF values for each word in vocabulary
        self.idf_values = []
        # Regular expression to match common punctuation marks
        self.punctuation_pattern = re.compile(r"[.,;:!?]")
    
    def preprocess_text(self, text):
        """Clean and normalize text"""
        # Convert non-string inputs to strings
        if not isinstance(text, str):
            text = str(text)
        
        # Convert all characters to lowercase for consistency
        text = text.lower()
        # Remove punctuation marks defined in our pattern
        text = self.punctuation_pattern.sub("", text)
        
        # Replace multiple whitespace characters with a single space
        # and trim leading/trailing whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def tokenize(self, text):
        """Split text into tokens"""
        # Simple whitespace tokenization - converts text to list of words
        return text.split()
    
    def build_vocabulary(self, documents):
        """Create vocabulary mapping from preprocessed documents"""
        # Set to collect all unique words from all documents
        unique_words = set()
        
        # Process each document to collect unique words
        for doc in documents:
            tokens = self.tokenize(doc)
            # Update set with new tokens (duplicates are automatically ignored)
            unique_words.update(tokens)
        
        # Create dictionary mapping each word to an index position
        # Words are sorted alphabetically to ensure consistent ordering
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(unique_words))}
        return self.vocabulary
    
    def calculate_tf(self, documents):
        """Calculate Term Frequency for each document"""
        # List to store TF vectors for all documents
        tf_matrix = []
        
        for doc in documents:
            # Split document into words
            tokens = self.tokenize(doc)
            # Initialize a zero vector with length equal to vocabulary size
            tf_vector = [0] * len(self.vocabulary)
            
            # Count occurrences of each word in the document
            for token in tokens:
                if token in self.vocabulary:
                    # Increment count for the word's index position
                    tf_vector[self.vocabulary[token]] += 1
            
            # Normalize frequencies by document length (words count)
            # to account for different document lengths
            total_terms = len(tokens)
            if total_terms > 0:  # Avoid division by zero
                tf_vector = [freq / total_terms for freq in tf_vector]
            
            # Add the document's TF vector to the matrix
            tf_matrix.append(tf_vector)
        
        return tf_matrix
    
    def calculate_idf(self, documents):
        """Calculate Inverse Document Frequency"""
        # Count total number of documents
        n_docs = len(documents)
        # Counter to track how many documents contain each word
        word_doc_count = Counter()
        
        # Count documents containing each unique word
        for doc in documents:
            # Get unique words in this document (using set to count each word once per doc)
            unique_words = set(self.tokenize(doc))
            for word in unique_words:
                # Increment document frequency for this word
                word_doc_count[word] += 1
        
        # Calculate IDF values for each word in vocabulary
        idf_vector = [0] * len(self.vocabulary)
        for word, idx in self.vocabulary.items():
            # Get number of documents containing this word
            doc_freq = word_doc_count[word]
            # Apply smoothed IDF formula: 1 + log((N+1)/(df+1))
            # Adding 1 prevents division by zero and log(0)
            idf_vector[idx] = 1 + np.log((n_docs + 1) / (doc_freq + 1))
        
        # Store IDF values for later use
        self.idf_values = idf_vector
        return idf_vector
    
    def normalize_vector(self, vector):
        """L2 normalize a vector"""
        # Calculate L2 norm (Euclidean length) of the vector
        norm = np.sqrt(sum(x**2 for x in vector))
        # Divide each element by the norm if it's not zero
        if norm > 0:
            return [x / norm for x in vector]
        # Return original vector if norm is zero
        return vector
    
    def fit_transform(self, documents):
        """Fit the vectorizer and transform documents to TF-IDF vectors"""
        # Clean and standardize all input documents
        cleaned_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Create vocabulary from all preprocessed documents
        self.build_vocabulary(cleaned_docs)
        
        # Calculate term frequency for each document
        tf_matrix = self.calculate_tf(cleaned_docs)
        # Calculate inverse document frequency for the corpus
        idf_vector = self.calculate_idf(cleaned_docs)
        
        # Calculate TF-IDF by multiplying TF and IDF values
        tfidf_matrix = []
        for tf_vector in tf_matrix:
            # Element-wise multiplication of TF and IDF
            tfidf_vector = [tf * idf for tf, idf in zip(tf_vector, idf_vector)]
            # Normalize the TF-IDF vector for better comparison between documents
            tfidf_vector = self.normalize_vector(tfidf_vector)
            tfidf_matrix.append(tfidf_vector)
        
        return tfidf_matrix
    
    def get_feature_names(self):
        """Return the vocabulary as a list of feature names"""
        # Return words in the vocabulary sorted by their index values
        return sorted(self.vocabulary.keys())

# Example usage
if __name__ == "__main__":
    # Sample documents for demonstration
    docs = [
        "Tom plays soccer!",
        "Tom loves basketball.",
        "Basketball is his hobby?",
        "Sarah loves basketball;"
    ]
    
    # Initialize and apply TF-IDF vectorization
    vectorizer = TFIDFVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    # Display results
    feature_names = vectorizer.get_feature_names()
    print("Vocabulary:", feature_names)
    print("\nTF-IDF Matrix:")
    for i, vector in enumerate(tfidf_matrix):
        # Round values to 4 decimal places for readability
        print(f"Document {i+1}: {[round(x, 4) for x in vector]}")
    
    # Show specific word contributions to each document vector
    print("\nNon-zero features by document:")
    for i, (doc, vector) in enumerate(zip(docs, tfidf_matrix)):
        print(f"\nDocument {i+1}: '{doc}'")
        for j, score in enumerate(vector):
            # Only show words with non-zero TF-IDF scores
            if score > 0:
                print(f"  {feature_names[j]}: {score:.4f}")
