# Imports
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import pickle
import os
from tqdm import tqdm
import re
import requests
import tarfile
import glob

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hyperparameters
embedding_dim = 128  # Embedding dimension
hidden_size = 256
num_layers = 2
sequence_length = 100  # Length of input sequences
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Character-level text processing functions
def get_shakespeare_text():
    """Download Shakespeare text for character-level modeling"""
    if os.path.exists('shakespeare.txt'):
        with open('shakespeare.txt', 'r', encoding='utf-8') as f:
            return f.read()
    
    print("Downloading Shakespeare text...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    try:
        response = requests.get(url)
        text = response.text
        
        with open('shakespeare.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Downloaded {len(text)} characters")
        return text
    except Exception as e:
        print(f"Failed to download: {e}")
        # Use a simple fallback text
        text = """To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.""" * 100
        return text

def get_sample_text():
    """Get a sample text for character-level modeling"""
    return """Hello world! This is a simple text for character-level language modeling.
We will train our RNN to generate text character by character.
The model will learn patterns in the text and generate new sequences.
This is a fundamental task in natural language processing.
Character-level models can generate creative and interesting text.
Let's see how well our model can learn these patterns!""" * 50

# Character-level text preprocessing class
class CharacterPreprocessor:
    def __init__(self):
        self.chars = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = 0
        
    def build_vocab(self, text):
        """Build character vocabulary from text"""
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(self.chars)}")
        
    def text_to_sequence(self, text):
        """Convert text to sequence of character indices"""
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def sequence_to_text(self, sequence):
        """Convert sequence of indices back to text"""
        return ''.join([self.idx_to_char[idx] for idx in sequence])

# Custom Dataset class for character-level text generation
class CharDataset(Dataset):
    def __init__(self, text, preprocessor, sequence_length):
        self.text = text
        self.preprocessor = preprocessor
        self.sequence_length = sequence_length
        self.data = preprocessor.text_to_sequence(text)
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        # Input sequence
        input_seq = self.data[idx:idx + self.sequence_length]
        # Target is the next character
        target = self.data[idx + self.sequence_length]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# Load and preprocess character-level data
def load_char_data():
    """Load text data for character-level modeling"""
    print("Loading text data...")
    
    # Try to get Shakespeare text, fallback to sample text
    try:
        text = get_shakespeare_text()
    except:
        print("Using sample text...")
        text = get_sample_text()
    
    print(f"Text length: {len(text)} characters")
    
    # Build character vocabulary
    preprocessor = CharacterPreprocessor()
    preprocessor.build_vocab(text)
    
    # Split text into train/test (90/10 split)
    split_idx = int(0.9 * len(text))
    train_text = text[:split_idx]
    test_text = text[split_idx:]
    
    print(f"Training text length: {len(train_text)}")
    print(f"Test text length: {len(test_text)}")
    
    return train_text, test_text, preprocessor

# Character-level RNN for text generation
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Forward propagate RNN
        out, hidden = self.rnn(embedded, hidden)  # (batch_size, seq_length, hidden_size)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Use only the last output for prediction
        out = out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(out)  # (batch_size, vocab_size)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

# Character-level GRU for text generation
class CharGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(CharGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        # Embedding layer
        embedded = self.embedding(x)
        
        # Forward propagate GRU
        out, hidden = self.gru(embedded, hidden)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Use only the last output for prediction
        out = out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(out)  # (batch_size, vocab_size)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

# Character-level LSTM for text generation
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        # Embedding layer
        embedded = self.embedding(x)
        
        # Forward propagate LSTM
        out, hidden = self.lstm(embedded, hidden)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Use only the last output for prediction
        out = out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(out)  # (batch_size, vocab_size)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)

# Load data
train_text, test_text, preprocessor = load_char_data()

# Create datasets
train_dataset = CharDataset(train_text, preprocessor, sequence_length)
test_dataset = CharDataset(test_text, preprocessor, sequence_length)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Vocabulary size: {preprocessor.vocab_size}")

# Initialize network (try CharRNN, CharGRU, or CharLSTM)
model = CharLSTM(preprocessor.vocab_size, embedding_dim, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Model: {model.__class__.__name__}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Train Network
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for batch_idx, (data, targets) in enumerate(progress_bar):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward pass
        outputs, _ = model(data)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        
        # Update weights
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

    print(f'Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {100.*correct/total:.2f}%')

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(device=device)
            y = y.to(device=device)

            outputs, _ = model(x)
            _, predictions = outputs.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    return num_correct / num_samples

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}%")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}%")

# Text generation functions
def generate_text(model, preprocessor, start_string="T", length=200, temperature=1.0):
    """Generate text using the trained model"""
    model.eval()
    
    # Convert start string to indices
    input_seq = [preprocessor.char_to_idx.get(ch, 0) for ch in start_string]
    generated = start_string
    
    # Initialize hidden state
    hidden = model.init_hidden(1)
    
    with torch.no_grad():
        # Process the start string
        for char_idx in input_seq[:-1]:
            input_tensor = torch.tensor([[char_idx]], dtype=torch.long).to(device)
            output, hidden = model(input_tensor, hidden)
        
        # Use the last character as input for generation
        input_tensor = torch.tensor([[input_seq[-1]]], dtype=torch.long).to(device)
        
        # Generate characters one by one
        for _ in range(length):
            output, hidden = model(input_tensor, hidden)
            
            # Apply temperature to the output distribution
            output = output.squeeze(0) / temperature
            probabilities = F.softmax(output, dim=0)
            
            # Sample from the distribution
            char_idx = torch.multinomial(probabilities, 1).item()
            char = preprocessor.idx_to_char[char_idx]
            
            generated += char
            
            # Use this character as input for the next iteration
            input_tensor = torch.tensor([[char_idx]], dtype=torch.long).to(device)
    
    return generated

def generate_text_greedy(model, preprocessor, start_string="T", length=200):
    """Generate text using greedy decoding (always pick most likely character)"""
    model.eval()
    
    # Convert start string to indices
    input_seq = [preprocessor.char_to_idx.get(ch, 0) for ch in start_string]
    generated = start_string
    
    # Initialize hidden state
    hidden = model.init_hidden(1)
    
    with torch.no_grad():
        # Process the start string
        for char_idx in input_seq[:-1]:
            input_tensor = torch.tensor([[char_idx]], dtype=torch.long).to(device)
            output, hidden = model(input_tensor, hidden)
        
        # Use the last character as input for generation
        input_tensor = torch.tensor([[input_seq[-1]]], dtype=torch.long).to(device)
        
        # Generate characters one by one
        for _ in range(length):
            output, hidden = model(input_tensor, hidden)
            
            # Pick the most likely character (greedy)
            char_idx = output.argmax().item()
            char = preprocessor.idx_to_char[char_idx]
            
            generated += char
            
            # Use this character as input for the next iteration
            input_tensor = torch.tensor([[char_idx]], dtype=torch.long).to(device)
    
    return generated

# Test text generation
print("\n" + "="*50)
print("TEXT GENERATION EXAMPLES")
print("="*50)

# Generate text with different start strings and temperatures
start_strings = ["The", "To be", "Hello", "I"]
temperatures = [0.5, 1.0, 1.5]

for start_string in start_strings:
    print(f"\nStarting with: '{start_string}'")
    print("-" * 40)
    
    # Greedy generation
    greedy_text = generate_text_greedy(model, preprocessor, start_string, length=150)
    print(f"Greedy: {greedy_text}")
    
    # Generate with different temperatures
    for temp in temperatures:
        generated_text = generate_text(model, preprocessor, start_string, length=150, temperature=temp)
        print(f"Temp {temp}: {generated_text}")
    print()

# Function to save the model
def save_model(model, preprocessor, filepath='char_rnn_model.pth'):
    """Save the trained model and preprocessor"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'preprocessor': preprocessor,
        'model_class': model.__class__.__name__,
        'vocab_size': preprocessor.vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_size': hidden_size,
        'num_layers': num_layers
    }, filepath)
    print(f"Model saved to {filepath}")

# Function to load the model
def load_model(filepath='char_rnn_model.pth'):
    """Load a saved model and preprocessor"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Recreate the model
    model_class = checkpoint['model_class']
    if model_class == 'CharLSTM':
        model = CharLSTM(checkpoint['vocab_size'], checkpoint['embedding_dim'], 
                        checkpoint['hidden_size'], checkpoint['num_layers'])
    elif model_class == 'CharGRU':
        model = CharGRU(checkpoint['vocab_size'], checkpoint['embedding_dim'], 
                       checkpoint['hidden_size'], checkpoint['num_layers'])
    else:
        model = CharRNN(checkpoint['vocab_size'], checkpoint['embedding_dim'], 
                       checkpoint['hidden_size'], checkpoint['num_layers'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    preprocessor = checkpoint['preprocessor']
    
    print(f"Model loaded from {filepath}")
    return model, preprocessor

# Save the trained model
save_model(model, preprocessor)

print("\n" + "="*50)
print("Training completed! You can now generate text using the model.")
print("Try different start strings and temperatures to see various outputs.")
print("="*50)