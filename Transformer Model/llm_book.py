"""
LLM from Scratch Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import re
import os
import urllib.request
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =============================================================================
# Chapter 2: Text Preprocessing and Tokenization
# =============================================================================

class SimpleTokenizerV1:
    """Simple tokenizer for basic text processing"""
    
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r"([,:;?_!\"()\']|--|\s)", text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = ' '.join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    """Enhanced tokenizer with unknown token handling"""
    
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int else "<|unk|>"
                       for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = ' '.join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


class GPTDatasetV1(Dataset):
    """Dataset class for GPT training with sliding window"""
    
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        # Use sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """Create dataloader for training"""
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
         dataset,
         batch_size=batch_size,
         shuffle=shuffle,
         drop_last=drop_last, 
         num_workers=num_workers 
         )
    return dataloader


# =============================================================================
# Chapter 3: Attention Mechanisms
# =============================================================================

class SelfAttention_v1(nn.Module):
    """Basic self-attention mechanism"""
    
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    """Enhanced self-attention with Linear layers"""
    
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    """Causal self-attention with masking"""
    
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    """Multi-head attention wrapper using multiple CausalAttention heads"""
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """Efficient multi-head attention implementation"""
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape for multi-head attention
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose for attention computation
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)
        
        # Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


# =============================================================================
# Chapter 4: GPT Model Architecture
# =============================================================================

class LayerNorm(nn.Module):
    """Layer normalization implementation"""
    
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU activation function"""
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Feed-forward network component"""
    
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )
    
    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers"""
    
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        # First sub-layer: Multi-head attention
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        # Second sub-layer: Feed-forward
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Residual connection

        return x


class GPTModel(nn.Module):
    """Main GPT model implementation"""
    
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# =============================================================================
# Configuration and Utilities
# =============================================================================

# GPT-2 124M parameter configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Simple text generation function"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx


def download_and_load_text(url, filename):
    """Download text file if not exists and load it"""
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def create_simple_tokenizer(text):
    """Create a simple tokenizer from text"""
    # Tokenize the text
    preprocessed = re.split(r"([,:;?_!\"()\']|--|\s)", text)
    preprocessed = [item for item in preprocessed if item.strip()]
    
    # Create vocabulary
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    
    return SimpleTokenizerV2(vocab)


def print_model_info(model):
    """Print model information"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    
    # Calculate size in MB (assuming float32)
    total_size_mb = total_params * 4 / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")


# =============================================================================
# Example Usage and Testing
# =============================================================================

def main():
    """Main function to demonstrate the LLM implementation"""
    
    # Set random seed for reproducibility
    torch.manual_seed(123)
    
    # Download sample text
    text_url = ("https://raw.githubusercontent.com/rasbt/"
               "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
               "the-verdict.txt")
    
    try:
        raw_text = download_and_load_text(text_url, 'the-verdict.txt')
        print(f"Loaded text with {len(raw_text)} characters")
    except Exception as e:
        print(f"Error loading text: {e}")
        # Use a simple example text
        raw_text = "Hello, world. This is a test of the LLM implementation."
    
    # Create tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    
    # Create model
    model = GPTModel(GPT_CONFIG_124M)
    print_model_info(model)
    
    # Example text generation
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    print(f"\nInput text: '{start_context}'")
    print(f"Encoded: {encoded}")
    
    # Generate text
    model.eval()
    output = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    
    decoded_text = tokenizer.decode(output.squeeze(0).tolist())
    print(f"Generated text: '{decoded_text}'")
    
    # Create dataloader example
    dataloader = create_dataloader_v1(
        raw_text[:1000],  # Use first 1000 characters for demo
        batch_size=2,
        max_length=64,
        stride=32,
        shuffle=False
    )
    
    # Show one batch
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print(f"\nDataloader batch:")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    
    # Test attention mechanism
    print(f"\nTesting attention mechanism...")
    x = torch.rand(2, 10, 768)  # [batch_size, seq_len, emb_dim]
    attention = MultiHeadAttention(
        d_in=768,
        d_out=768,
        context_length=1024,
        dropout=0.1,
        num_heads=12
    )
    
    attn_output = attention(x)
    print(f"Attention input shape: {x.shape}")
    print(f"Attention output shape: {attn_output.shape}")
    
    print("\nLLM implementation complete!")


if __name__ == "__main__":
    main()