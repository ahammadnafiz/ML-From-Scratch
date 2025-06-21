import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from collections import Counter
import re
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in the original Transformer paper.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Calculate div_term for the sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism as described in the Transformer paper.
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V shape: (batch_size, n_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        output = self.W_o(attention_output)
        return output

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """
    Single encoder layer with self-attention and feed-forward network.
    Includes residual connections and layer normalization.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    """
    Single decoder layer with masked self-attention, encoder-decoder attention,
    and feed-forward network. Includes residual connections and layer normalization.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention with residual connection and layer norm
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Encoder-decoder attention with residual connection and layer norm
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Encoder(nn.Module):
    """
    Transformer encoder consisting of a stack of encoder layers.
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x

class Decoder(nn.Module):
    """
    Transformer decoder consisting of a stack of decoder layers.
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x

class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, 
                 n_heads: int = 8, n_layers: int = 6, d_ff: int = 2048, max_len: int = 5000):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_heads, n_layers, d_ff, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads, n_layers, d_ff, max_len)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self.init_parameters()
        
    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src):
        # Create mask for padding tokens (assuming 0 is padding)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        # Create mask for padding tokens
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        # Create causal mask to prevent looking at future tokens
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool().to(tgt.device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.linear(decoder_output)
        
        return output

# Custom vocabulary class for handling tokens
class Vocabulary:
    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.idx = 0
        
        # Add special tokens
        self.add_token('<pad>')
        self.add_token('<sos>')
        self.add_token('<eos>')
        self.add_token('<unk>')
        
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
    
    def add_token(self, token):
        if token not in self.token_to_idx:
            self.token_to_idx[token] = self.idx
            self.idx_to_token[self.idx] = token
            self.idx += 1
    
    def __len__(self):
        return len(self.token_to_idx)
    
    def encode(self, tokens):
        return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]
    
    def decode(self, indices):
        return [self.idx_to_token.get(idx, '<unk>') for idx in indices]

def tokenize_simple(text: str) -> List[str]:
    """Simple tokenization by splitting on whitespace and punctuation."""
    # Convert to lowercase and split on whitespace and common punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    return tokens

def build_vocab_from_dataset(dataset, min_freq=2):
    """Build vocabularies for source and target languages from the dataset."""
    en_vocab = Vocabulary()
    es_vocab = Vocabulary()
    
    # Count tokens
    en_counter = Counter()
    es_counter = Counter()
    
    print("Building vocabularies...")
    for example in dataset:
        # Handle the opus_books format where translations are in 'translation' field
        # with 'en' and 'es' keys directly
        en_tokens = tokenize_simple(example['translation']['en'])
        es_tokens = tokenize_simple(example['translation']['es'])
        
        en_counter.update(en_tokens)
        es_counter.update(es_tokens)
    
    # Add tokens that meet minimum frequency
    for token, freq in en_counter.items():
        if freq >= min_freq:
            en_vocab.add_token(token)
    
    for token, freq in es_counter.items():
        if freq >= min_freq:
            es_vocab.add_token(token)
    
    print(f"English vocabulary size: {len(en_vocab)}")
    print(f"Spanish vocabulary size: {len(es_vocab)}")
    
    return en_vocab, es_vocab

class HuggingFaceTranslationDataset(Dataset):
    """Dataset wrapper for HuggingFace translation datasets."""
    
    def __init__(self, hf_dataset, en_vocab, es_vocab, max_len=128):
        self.hf_dataset = hf_dataset
        self.en_vocab = en_vocab
        self.es_vocab = es_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        en_text = example['translation']['en']
        es_text = example['translation']['es']
        
        # Tokenize
        en_tokens = tokenize_simple(en_text)
        es_tokens = tokenize_simple(es_text)
        
        # Limit length
        en_tokens = en_tokens[:self.max_len-2]  # Reserve space for SOS/EOS
        es_tokens = es_tokens[:self.max_len-2]
        
        # Convert to IDs with SOS/EOS tokens
        en_ids = [self.en_vocab.sos_idx] + self.en_vocab.encode(en_tokens) + [self.en_vocab.eos_idx]
        es_ids = [self.es_vocab.sos_idx] + self.es_vocab.encode(es_tokens) + [self.es_vocab.eos_idx]
        
        return torch.tensor(en_ids), torch.tensor(es_ids)

def collate_fn(batch):
    """Collate function to pad sequences to same length."""
    en_batch, es_batch = zip(*batch)
    
    # Pad sequences
    en_batch = torch.nn.utils.rnn.pad_sequence(en_batch, batch_first=True, padding_value=0)
    es_batch = torch.nn.utils.rnn.pad_sequence(es_batch, batch_first=True, padding_value=0)
    
    return en_batch, es_batch

def train_model(model, dataloader, optimizer, criterion, num_epochs=10):
    """Train the transformer model."""
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            
            # Prepare input and target for decoder
            tgt_input = tgt[:, :-1]  # Remove last token for input
            tgt_output = tgt[:, 1:]  # Remove first token for target
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, tgt_input)
            
            # Calculate loss
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}')
    
    return losses

def translate(model, src_text, en_vocab, es_vocab, max_len=50):
    """Translate a source text using the trained model."""
    model.eval()
    
    with torch.no_grad():
        # Tokenize and convert to IDs
        src_tokens = tokenize_simple(src_text)
        src_ids = [en_vocab.sos_idx] + en_vocab.encode(src_tokens) + [en_vocab.eos_idx]
        src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
        
        # Start with SOS token
        tgt_ids = [es_vocab.sos_idx]
        
        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)
            
            # Get model output
            output = model(src_tensor, tgt_tensor)
            
            # Get the last token's prediction
            next_token_logits = output[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            tgt_ids.append(next_token_id)
            
            # Stop if EOS token is generated
            if next_token_id == es_vocab.eos_idx:
                break
        
        # Convert back to text
        translated_tokens = es_vocab.decode(tgt_ids[1:-1])  # Remove SOS and EOS
        translated_text = ' '.join(translated_tokens)
        return translated_text

def evaluate_model(model, test_dataloader, en_vocab, es_vocab, num_examples=10):
    """Evaluate the model on test data."""
    model.eval()
    
    print("\nEvaluating model on test data:")
    print("="*50)
    
    with torch.no_grad():
        for i, (src_batch, tgt_batch) in enumerate(test_dataloader):
            if i >= num_examples:
                break
                
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            # Take first example from batch
            src_ids = src_batch[0].cpu().tolist()
            tgt_ids = tgt_batch[0].cpu().tolist()
            
            # Decode source and reference
            src_tokens = en_vocab.decode(src_ids)
            tgt_tokens = es_vocab.decode(tgt_ids)
            
            src_text = ' '.join([t for t in src_tokens if t not in ['<pad>', '<sos>', '<eos>']])
            ref_text = ' '.join([t for t in tgt_tokens if t not in ['<pad>', '<sos>', '<eos>']])
            
            # Generate translation
            pred_text = translate(model, src_text, en_vocab, es_vocab)
            
            print(f"Example {i+1}:")
            print(f"Source:     {src_text}")
            print(f"Reference:  {ref_text}")
            print(f"Predicted:  {pred_text}")
            print("-" * 50)

if __name__ == "__main__":
    # Load dataset from HuggingFace
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("Helsinki-NLP/opus_books", "en-es", split="train")
    
    # Use a subset for faster training (you can increase this)
    dataset = dataset.select(range(min(10000, len(dataset))))
    
    # Split into train and test
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    
    # Build vocabularies
    en_vocab, es_vocab = build_vocab_from_dataset(train_dataset, min_freq=2)
    
    # Create dataset objects
    train_torch_dataset = HuggingFaceTranslationDataset(train_dataset, en_vocab, es_vocab)
    test_torch_dataset = HuggingFaceTranslationDataset(test_dataset, en_vocab, es_vocab)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_torch_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_torch_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Model parameters
    src_vocab_size = len(en_vocab)
    tgt_vocab_size = len(es_vocab)
    d_model = 256  # Reasonable size
    n_heads = 8
    n_layers = 4   # Moderate number of layers
    d_ff = 512
    
    # Create model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    
    print("Starting training...")
    losses = train_model(model, train_dataloader, optimizer, criterion, num_epochs=100)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Evaluate the model
    evaluate_model(model, test_dataloader, en_vocab, es_vocab, num_examples=10)
    
    # Test some custom sentences
    print("\nTesting custom translations:")
    test_sentences = [
        "hello world",
        "how are you today",
        "i love machine learning",
        "the weather is beautiful",
        "thank you very much"
    ]
    
    for sentence in test_sentences:
        translation = translate(model, sentence, en_vocab, es_vocab)
        print(f"English: {sentence}")
        print(f"Spanish: {translation}")
        print("-" * 30)