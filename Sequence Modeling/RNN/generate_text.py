# Load and use trained character RNN model for text generation
import torch
import torch.nn.functional as F
from torch import nn
import os

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Character-level LSTM for text generation (same as in training script)
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

# Character-level text preprocessing class (needed for loading saved model)
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

def load_model(filepath='char_rnn_model.pth'):
    """Load a saved model and preprocessor"""
    if not os.path.exists(filepath):
        print(f"Model file {filepath} not found!")
        return None, None
    
    print(f"Loading model from {filepath}...")
    
    # Load with weights_only=False for compatibility with older PyTorch versions
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
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
    model.eval()  # Set to evaluation mode
    
    preprocessor = checkpoint['preprocessor']
    
    print(f"Model loaded successfully!")
    print(f"Model type: {model_class}")
    print(f"Vocabulary size: {preprocessor.vocab_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model, preprocessor

def generate_text(model, preprocessor, start_string="The", length=200, temperature=1.0):
    """Generate text using the trained model"""
    model.eval()
    
    # Convert start string to indices
    input_seq = [preprocessor.char_to_idx.get(ch, 0) for ch in start_string]
    generated = start_string
    
    # Initialize hidden state
    hidden = model.init_hidden(1)
    
    with torch.no_grad():
        # Process the start string (if longer than 1 character)
        if len(input_seq) > 1:
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

def generate_text_greedy(model, preprocessor, start_string="The", length=200):
    """Generate text using greedy decoding (always pick most likely character)"""
    model.eval()
    
    # Convert start string to indices
    input_seq = [preprocessor.char_to_idx.get(ch, 0) for ch in start_string]
    generated = start_string
    
    # Initialize hidden state
    hidden = model.init_hidden(1)
    
    with torch.no_grad():
        # Process the start string (if longer than 1 character)
        if len(input_seq) > 1:
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

def interactive_generation(model, preprocessor):
    """Interactive text generation"""
    print("\n" + "="*60)
    print("INTERACTIVE TEXT GENERATION")
    print("="*60)
    print("Commands:")
    print("  - Type a seed text to generate from")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'help' for more options")
    print("-"*60)
    
    while True:
        try:
            user_input = input("\nEnter seed text (or command): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  quit/exit - Exit the program")
                print("  help - Show this help")
                print("\nGeneration options:")
                print("  Just type any text as a seed to generate from it")
                print("  The model will generate text based on the patterns it learned")
                continue
            elif user_input == '':
                continue
            
            print(f"\nGenerating text from seed: '{user_input}'")
            print("-" * 40)
            
            # Generate with different temperatures
            temperatures = [0.5, 1.0, 1.5]
            length = 150
            
            # Greedy generation
            greedy_text = generate_text_greedy(model, preprocessor, user_input, length)
            print(f"Greedy: {greedy_text}")
            print()
            
            # Temperature-based generation
            for temp in temperatures:
                generated_text = generate_text(model, preprocessor, user_input, length, temperature=temp)
                print(f"Temp {temp}: {generated_text}")
                print()
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    # Load the trained model
    model, preprocessor = load_model('char_rnn_model.pth')
    
    if model is None:
        print("Failed to load model. Please make sure 'char_rnn_model.pth' exists.")
        return
    
    print("\n" + "="*60)
    print("SHAKESPEARE TEXT GENERATOR")
    print("="*60)
    print(f"Available characters: {''.join(preprocessor.chars)}")
    print("-"*60)
    
    # Generate some sample texts
    print("\nSAMPLE GENERATIONS:")
    print("="*40)
    
    start_strings = [
        "ROMEO:",
        "JULIET:",
        "To be or not to be",
        "The",
        "What",
        "How",
        "When",
        "Where"
    ]
    
    temperatures = [0.7, 1.0, 1.2]
    
    for start_string in start_strings:
        print(f"\n🎭 Starting with: '{start_string}'")
        print("-" * 50)
        
        # Generate with different temperatures
        for temp in temperatures:
            generated_text = generate_text(model, preprocessor, start_string, length=200, temperature=temp)
            print(f"Temperature {temp}: {generated_text}")
        print()
    
    # Start interactive generation
    interactive_generation(model, preprocessor)

if __name__ == "__main__":
    main()
