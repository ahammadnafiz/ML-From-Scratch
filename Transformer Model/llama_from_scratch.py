
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
import random
import os
import sentencepiece as spm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataclasses import dataclass
import math

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# =============================================================================
# STEP 1: DATA PREPARATION AND PREPROCESSING
# =============================================================================

def prepare_data_splits(input_file, train_ratio=0.8, val_ratio=0.1):
    """
    Split the input text file into train, validation, and test sets.
    
    Args:
        input_file (str): Path to the input text file
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
    
    Returns:
        tuple: Paths to train, validation, and test files
    """
    print("📖 Preparing data splits...")
    
    # Define output file paths
    train_file = 'Meditations_train.txt'
    val_file = 'Meditations_val.txt'
    test_file = 'Meditations_test.txt'
    
    # Read and shuffle the input data
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    random.shuffle(lines)
    
    # Calculate split indices
    total = len(lines)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    # Split the data
    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]
    
    # Write split files
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
    
    print(f"✅ Data splits created:")
    print(f"   📊 Train size: {len(train_lines)} lines")
    print(f"   📊 Validation size: {len(val_lines)} lines")
    print(f"   📊 Test size: {len(test_lines)} lines")
    
    return train_file, val_file, test_file

def train_tokenizer(train_file, model_prefix='bpe_meditations', vocab_size=1000):
    """
    Train a SentencePiece tokenizer on the training data.
    
    Args:
        train_file (str): Path to training file
        model_prefix (str): Prefix for tokenizer model files
        vocab_size (int): Size of vocabulary
    
    Returns:
        str: Path to the trained tokenizer model
    """
    print("🔤 Training tokenizer...")
    
    spm.SentencePieceTrainer.train(
        input=train_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',  # Byte Pair Encoding
        character_coverage=1.0,  # Cover all characters in the input
        unk_id=0,  # Unknown token ID
        bos_id=1,  # Beginning of sentence token ID
        eos_id=2,  # End of sentence token ID
        pad_id=3   # Padding token ID
    )
    
    tokenizer_path = f"{model_prefix}.model"
    print(f"✅ Tokenizer trained and saved to: {tokenizer_path}")
    
    return tokenizer_path

# Prepare data and tokenizer
input_file = '/content/Meditations Train.txt'
train_file, val_file, test_file = prepare_data_splits(input_file)
tokenizer_model_path = train_tokenizer(train_file)

# =============================================================================
# STEP 2: CONFIGURATION AND DATA STRUCTURES
# =============================================================================

@dataclass
class LLMConfig:
    """
    Configuration class for LLaMA model hyperparameters.
    
    Model Architecture:
        - vocab_size: Size of vocabulary (number of unique tokens)
        - block_size: Maximum sequence length (context window)
        - n_layer: Number of transformer blocks
        - n_head: Number of attention heads
        - n_embd: Embedding dimension
        - dropout: Dropout probability for regularization
    
    Training Parameters:
        - batch_size: Number of sequences per batch
        - learning_rate: Learning rate for optimizer
        - weight_decay: L2 regularization strength
        - betas: Adam optimizer beta parameters
        - max_iters: Number of training epochs
    """
    # Model architecture
    vocab_size: int = 1000      # Vocabulary size from tokenizer
    block_size: int = 256       # Context length (sequence length)
    n_layer: int = 6           # Number of transformer layers
    n_head: int = 6            # Number of attention heads
    n_embd: int = 384          # Embedding dimension
    dropout: float = 0.1       # Dropout rate for regularization

    # Training hyperparameters
    batch_size: int = 32       # Reduced for stability
    learning_rate: float = 3e-4  # Learning rate
    weight_decay: float = 1e-1   # Weight decay for regularization
    betas: tuple = (0.9, 0.95)  # Adam optimizer betas

    # Training settings
    max_iters: int = 5         # Number of epochs (keep low for demo)
    save_checkpoint: bool = True
    out_dir: str = 'checkpoints/meditations-llm'
    
    # Generation settings
    generation_interval: int = 1  # Generate text every N epochs

class TokenizedTextDataset(Dataset):
    """
    Custom dataset class for tokenized text data.
    
    This dataset:
    1. Loads and tokenizes text using SentencePiece
    2. Creates sliding windows of text for training
    3. Provides input-target pairs where target is shifted by 1 token
    
    Args:
        file_path (str): Path to text file
        tokenizer_model_path (str): Path to trained tokenizer
        block_size (int): Sequence length for each sample
    """
    
    def __init__(self, file_path, tokenizer_model_path, block_size):
        print(f"📚 Loading dataset from: {file_path}")
        
        # Initialize tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_model_path)
        self.block_size = block_size

        # Read and tokenize text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Convert text to token IDs
        token_ids = self.sp.encode(text, out_type=int)
        self.data = torch.tensor(token_ids, dtype=torch.long)
        
        print(f"   📊 Total tokens: {len(self.data)}")
        print(f"   📊 Total samples: {len(self.data) - self.block_size}")

    def __len__(self):
        """Return number of possible sequences."""
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            x: Input sequence of length block_size
            y: Target sequence (x shifted by 1 position)
        """
        # Input sequence: tokens[idx:idx+block_size]
        x = self.data[idx : idx + self.block_size]
        # Target sequence: tokens[idx+1:idx+1+block_size] (shifted by 1)
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y

# =============================================================================
# STEP 3: MODEL COMPONENTS AND HELPER FUNCTIONS
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm is used in LLaMA instead of LayerNorm for better performance.
    It normalizes using the RMS (root mean square) and applies a learnable scale.
    
    Formula: RMSNorm(x) = (x / RMS(x)) * scale
    where RMS(x) = sqrt(mean(x^2))
    """
    
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps  # Small constant to prevent division by zero
        self.scale = nn.Parameter(torch.ones(dim))  # Learnable scale parameter

    def forward(self, x):
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor of shape (..., dim)
        
        Returns:
            Normalized tensor with same shape as input
        """
        # Calculate L2 norm along the last dimension
        norm = x.norm(dim=-1, keepdim=True)
        # Calculate RMS (root mean square)
        rms = norm / math.sqrt(x.size(-1))
        # Normalize and scale
        x_normed = x / (rms + self.eps)
        return self.scale * x_normed


def rotate_half(x):
    """
    Rotate the second half of the last dimension.
    
    This is used in rotary positional embeddings to create the rotation.
    
    Args:
        x: Input tensor (..., d) where d is even
    
    Returns:
        Tensor with second half rotated: [..., -x[d/2:], x[:d/2]]
    """
    x1, x2 = x.chunk(2, dim=-1)  # Split into two halves
    return torch.cat((-x2, x1), dim=-1)  # Rotate: (-x2, x1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary positional embeddings to query and key tensors.
    
    Rotary embeddings encode position information by rotating the embedding
    vectors in a way that preserves relative position information.
    
    Args:
        q: Query tensor (batch, n_head, seq_len, head_dim)
        k: Key tensor (batch, n_head, seq_len, head_dim)
        cos: Cosine values for rotation (1, 1, seq_len, head_dim)
        sin: Sine values for rotation (1, 1, seq_len, head_dim)
    
    Returns:
        Tuple of rotated (q, k) tensors
    """
    # Apply rotation: q_rot = q * cos + rotate_half(q) * sin
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def create_rotary_embeddings(seq_len, dim, device):
    """
    Create rotary positional embeddings (RoPE).
    
    RoPE encodes position by rotating embedding vectors. This allows the model
    to understand relative positions without explicit positional embeddings.
    
    Args:
        seq_len: Maximum sequence length
        dim: Embedding dimension (should be even)
        device: Device to create tensors on
    
    Returns:
        Tuple of (cos, sin) tensors for rotary embeddings
    """
    # Create inverse frequencies: 1 / (10000^(2i/dim)) for i in [0, dim/2)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    # Create position indices
    t = torch.arange(seq_len, device=device).type_as(inv_freq)
    
    # Calculate frequencies for each position
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, dim/2)
    
    # Duplicate frequencies for cos and sin pairs
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)

    # Create cos and sin embeddings with shape (1, 1, seq_len, dim)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    
    return cos, sin


class LLaMABlock(nn.Module):
    """
    A single LLaMA transformer block.
    
    Each block consists of:
    1. Multi-head self-attention with rotary positional embeddings
    2. SwiGLU feed-forward network
    3. RMSNorm for normalization (applied before attention and FFN)
    4. Residual connections around both attention and FFN
    
    Architecture:
        x -> RMSNorm -> MultiHeadAttention -> residual -> 
        x -> RMSNorm -> SwiGLU FFN -> residual -> output
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head  # Dimension per attention head
        
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

        # Pre-normalization layers (applied before attention and FFN)
        self.norm1 = RMSNorm(self.n_embd)
        self.norm2 = RMSNorm(self.n_embd)

        # Multi-head attention components
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)  # Combined Q, K, V projection
        self.attn_out = nn.Linear(self.n_embd, self.n_embd, bias=False)  # Output projection

        # SwiGLU Feed-Forward Network
        # SwiGLU: FFN(x) = (W1(x) * SiLU(W2(x))) @ W3
        self.ff1 = nn.Linear(self.n_embd, 4 * self.n_embd, bias=False)  # Expand to 4x dimension
        self.ff2 = nn.Linear(4 * self.n_embd // 2, self.n_embd, bias=False)  # Contract back (half dim after SwiGLU)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, mask=None):
        """
        Forward pass through the LLaMA block.
        
        Args:
            x: Input tensor (batch_size, seq_len, n_embd)
            cos: Cosine rotary embeddings (1, 1, seq_len, head_dim)
            sin: Sine rotary embeddings (1, 1, seq_len, head_dim)
            mask: Attention mask (1, 1, seq_len, seq_len)
        
        Returns:
            Output tensor with same shape as input
        """
        B, T, C = x.size()  # Batch, Time (sequence), Channels (embedding dim)

        # ===== MULTI-HEAD SELF-ATTENTION =====
        # Apply pre-normalization
        x_norm = self.norm1(x)
        
        # Compute Q, K, V in one go
        qkv = self.qkv(x_norm)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # Split into Q, K, V each (B, T, C)

        # Reshape for multi-head attention: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply rotary positional embeddings to Q and K
        q, k = apply_rotary_pos_emb(q, k, cos[:, :, :T, :], sin[:, :, :T, :])

        # Scaled dot-product attention
        # Attention scores: Q @ K^T / sqrt(head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask (prevent attending to future tokens)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: Attention @ V
        attn_output = attn_weights @ v  # (B, n_head, T, head_dim)
        
        # Reshape back: (B, n_head, T, head_dim) -> (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply output projection and residual connection
        x = x + self.attn_out(attn_output)

        # ===== SWIGLU FEED-FORWARD NETWORK =====
        # Apply pre-normalization
        x_norm = self.norm2(x)

        # SwiGLU activation: split the intermediate layer in half
        ff_intermediate = self.ff1(x_norm)  # (B, T, 4*C)
        x1, x2 = ff_intermediate.chunk(2, dim=-1)  # Split into two halves
        
        # Apply SwiGLU: x1 * SiLU(x2)
        ff_activated = x1 * F.silu(x2)  # SiLU(x) = x * sigmoid(x)
        
        # Project back to original dimension
        ff_output = self.ff2(ff_activated)
        ff_output = self.dropout(ff_output)

        # Apply residual connection
        x = x + ff_output

        return x


class LLaMA(nn.Module):
    """
    LLaMA (Large Language Model Meta AI) implementation.
    
    Architecture:
    1. Token embeddings (learned)
    2. Stack of LLaMA transformer blocks
    3. Final RMSNorm
    4. Output projection (tied with input embeddings)
    
    Key features:
    - Rotary Positional Embeddings (RoPE) instead of learned positional embeddings
    - RMSNorm instead of LayerNorm
    - SwiGLU activation in feed-forward networks
    - Weight tying between input and output embeddings
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding layer: converts token IDs to vectors
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Dimension for rotary embeddings (per attention head)
        self.pos_emb_dim = config.n_embd // config.n_head

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            LLaMABlock(config) for _ in range(config.n_layer)
        ])
        
        # Final normalization layer
        self.norm = RMSNorm(config.n_embd)

        # Output projection layer (language modeling head)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie input and output embeddings (common practice in language models)
        self.head.weight = self.token_emb.weight
        
        print(f"🏗️ LLaMA model initialized:")
        print(f"   📊 Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   📊 Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, idx):
        """
        Forward pass through the LLaMA model.
        
        Args:
            idx: Input token indices (batch_size, seq_len)
        
        Returns:
            Logits for next token prediction (batch_size, seq_len, vocab_size)
        """
        B, T = idx.size()  # Batch size, sequence length
        device = idx.device

        # Convert token IDs to embeddings
        x = self.token_emb(idx)  # (B, T, n_embd)

        # Create rotary positional embeddings (fix device placement issue)
        cos, sin = create_rotary_embeddings(T, self.pos_emb_dim, device)

        # Create causal attention mask (lower triangular)
        # This ensures tokens can only attend to previous tokens
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, mask)

        # Apply final normalization
        x = self.norm(x)

        # Project to vocabulary size for next token prediction
        logits = self.head(x)  # (B, T, vocab_size)
        
        return logits
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# =============================================================================
# STEP 4: EVALUATION AND TEXT GENERATION FUNCTIONS
# =============================================================================

@torch.no_grad()
def evaluate_loss(model, dataloader, device):
    """
    Evaluate the model on a dataset and return average loss.
    
    Args:
        model: The LLaMA model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
    
    Returns:
        Average loss per token
    """
    print("📊 Evaluating model...")
    model.eval()  # Set model to evaluation mode
    
    total_loss = 0.0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    # Evaluate on all batches
    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        logits = model(x)
        
        # Reshape for loss calculation
        logits_flat = logits.view(-1, logits.size(-1))  # (B*T, vocab_size)
        targets_flat = y.view(-1)  # (B*T,)

        # Calculate loss
        loss = loss_fn(logits_flat, targets_flat)
        total_loss += loss.item()
        total_tokens += targets_flat.ne(-100).sum().item()  # Count non-ignored tokens

    model.train()  # Set back to training mode
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    print(f"   ✅ Average loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def generate_text(model, tokenizer, device, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
    """
    Generate text using the trained model.
    
    Args:
        model: Trained LLaMA model
        tokenizer: SentencePiece tokenizer
        device: Device to run generation on
        prompt: Input text to start generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Only sample from top-k most likely tokens
    
    Returns:
        Generated text as string
    """
    model.eval()  # Set to evaluation mode
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"🎯 Generating text from prompt: '{prompt}'")
    print(f"   🔧 Settings: max_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}")

    # Generate tokens one by one
    for step in range(max_new_tokens):
        # Only use the last block_size tokens (context window)
        input_cond = input_ids[:, -model.config.block_size:]

        # Get model predictions
        logits = model(input_cond)
        logits = logits[:, -1, :] / temperature  # Get last token logits and apply temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits.scatter_(1, top_k_indices, top_k_values)
            logits = filtered_logits

        # Convert to probabilities and sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        input_ids = torch.cat((input_ids, next_token), dim=1)

    # Decode the full sequence
    output = tokenizer.decode(input_ids[0].tolist())
    model.train()  # Set back to training mode
    
    return output


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add minimum validation loss annotation
    min_val_idx = np.argmin(val_losses)
    min_val_loss = val_losses[min_val_idx]
    plt.annotate(f'Min Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_idx + 1}', 
                xy=(min_val_idx + 1, min_val_loss), 
                xytext=(10, 10), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training curve saved to: {save_path}")
    
    plt.show()


def print_model_info(model, config):
    """Print detailed model information."""
    total_params, trainable_params = model.count_parameters()
    
    print("\n" + "="*60)
    print("🏗️  MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    print(f"📊 Model Type: LLaMA")
    print(f"📊 Vocabulary Size: {config.vocab_size:,}")
    print(f"📊 Context Length: {config.block_size}")
    print(f"📊 Embedding Dimension: {config.n_embd}")
    print(f"📊 Number of Layers: {config.n_layer}")
    print(f"📊 Number of Attention Heads: {config.n_head}")
    print(f"📊 Head Dimension: {config.n_embd // config.n_head}")
    print(f"📊 Dropout Rate: {config.dropout}")
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"📊 Trainable Parameters: {trainable_params:,}")
    print(f"📊 Model Size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # Assuming float32
    print("="*60)

# =============================================================================
# STEP 5: ENHANCED TRAINING FUNCTION
# =============================================================================

def train_model(model, train_loader, val_loader, config, device, tokenizer=None):
    """
    Train the LLaMA model with detailed logging and text generation.
    
    Args:
        model: LLaMA model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        tokenizer: Optional tokenizer for text generation during training
    
    Returns:
        Tuple of (train_losses, val_losses) for plotting
    """
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    
    # Move model to device
    model.to(device)
    print(f"📱 Training on device: {device}")
    
    # Setup optimizer with detailed info
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        betas=config.betas, 
        weight_decay=config.weight_decay
    )
    
    print(f"🔧 Optimizer: AdamW")
    print(f"   📊 Learning Rate: {config.learning_rate}")
    print(f"   📊 Weight Decay: {config.weight_decay}")
    print(f"   📊 Betas: {config.betas}")
    
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)
    
    # Tracking variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Sample prompts for generation during training
    sample_prompts = [
        "The meaning of life is",
        "To be happy, one must",
        "Philosophy teaches us",
        "A wise person"
    ]
    
    print(f"\n🎯 Training for {config.max_iters} epochs...")
    print(f"📊 Batch size: {config.batch_size}")
    print(f"📊 Batches per epoch: {len(train_loader)}")
    
    # Training loop
    for epoch in range(config.max_iters):
        epoch_start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"📅 EPOCH {epoch + 1}/{config.max_iters}")
        print(f"{'='*50}")
        
        # =============================================================================
        # TRAINING PHASE
        # =============================================================================
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"🏋️ Training Epoch {epoch+1}")
        
        for batch_idx, (x, y) in enumerate(train_pbar):
            # Move data to device
            x = x.to(device)
            y = y.to(device)

            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(x)
            
            # Calculate loss
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = y.view(-1)
            loss = loss_fn(logits_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Track loss
            total_train_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_train_loss/num_batches:.4f}'
            })
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # =============================================================================
        # VALIDATION PHASE
        # =============================================================================
        print("📊 Evaluating on validation set...")
        val_loss = evaluate_loss(model, val_loader, device)
        val_losses.append(val_loss)
        
        # =============================================================================
        # EPOCH SUMMARY
        # =============================================================================
        epoch_time = time.time() - epoch_start_time
        
        print(f"\n📈 Epoch {epoch+1} Results:")
        print(f"   🏋️ Training Loss: {avg_train_loss:.4f}")
        print(f"   📊 Validation Loss: {val_loss:.4f}")
        print(f"   ⏱️ Epoch Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss and config.save_checkpoint:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config.out_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"   💾 Saved best model (val_loss: {val_loss:.4f}) to: {checkpoint_path}")
        
        # =============================================================================
        # TEXT GENERATION DURING TRAINING
        # =============================================================================
        if tokenizer and (epoch + 1) % config.generation_interval == 0:
            print(f"\n🎨 Generating text samples after epoch {epoch+1}:")
            print("-" * 50)
            
            for i, prompt in enumerate(sample_prompts):
                try:
                    generated = generate_text(
                        model, tokenizer, device, prompt, 
                        max_new_tokens=30, temperature=0.8, top_k=40
                    )
                    print(f"💭 Prompt {i+1}: '{prompt}'")
                    print(f"📝 Generated: {generated}")
                    print()
                except Exception as e:
                    print(f"❌ Generation failed for prompt '{prompt}': {e}")
    
    # =============================================================================
    # TRAINING COMPLETE
    # =============================================================================
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED!")
    print("="*60)
    print(f"🏆 Best Validation Loss: {best_val_loss:.4f}")
    print(f"📈 Final Training Loss: {train_losses[-1]:.4f}")
    print(f"📊 Final Validation Loss: {val_losses[-1]:.4f}")
    
    # Plot training curves
    plot_path = os.path.join(config.out_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, save_path=plot_path)
    
    return train_losses, val_losses

# =============================================================================
# STEP 6: MAIN EXECUTION AND TESTING
# =============================================================================

def main():
    """Main function to run the complete LLaMA training pipeline."""
    
    print("\n" + "="*60)
    print("🦙 LLAMA FROM SCRATCH - TRAINING PIPELINE")
    print("="*60)
    
    # Initialize configuration
    config = LLMConfig()
    
    print("⚙️ Configuration:")
    print(f"   📊 Vocabulary Size: {config.vocab_size}")
    print(f"   📊 Context Length: {config.block_size}")
    print(f"   📊 Model Layers: {config.n_layer}")
    print(f"   📊 Attention Heads: {config.n_head}")
    print(f"   📊 Embedding Dim: {config.n_embd}")
    print(f"   📊 Batch Size: {config.batch_size}")
    print(f"   📊 Learning Rate: {config.learning_rate}")
    print(f"   📊 Max Epochs: {config.max_iters}")
    
    # =============================================================================
    # SETUP DATASETS AND LOADERS
    # =============================================================================
    print("\n📚 Setting up datasets...")
    
    # Create datasets
    train_dataset = TokenizedTextDataset(train_file, tokenizer_model_path, config.block_size)
    val_dataset = TokenizedTextDataset(val_file, tokenizer_model_path, config.block_size)
    test_dataset = TokenizedTextDataset(test_file, tokenizer_model_path, config.block_size)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✅ Datasets created:")
    print(f"   📊 Training samples: {len(train_dataset)}")
    print(f"   📊 Validation samples: {len(val_dataset)}")
    print(f"   📊 Test samples: {len(test_dataset)}")
    
    # =============================================================================
    # INITIALIZE MODEL AND DEVICE
    # =============================================================================
    print("\n🏗️ Initializing model...")
    
    # Initialize model
    model = LLaMA(config)
    
    # Print model information
    print_model_info(model, config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Using device: {device}")
    if torch.cuda.is_available():
        print(f"   🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # =============================================================================
    # SETUP TOKENIZER FOR GENERATION
    # =============================================================================
    print("\n🔤 Loading tokenizer for text generation...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_model_path)
    print(f"✅ Tokenizer loaded with vocab size: {tokenizer.vocab_size()}")
    
    # =============================================================================
    # TRAIN THE MODEL
    # =============================================================================
    print("\n🚀 Starting training...")
    
    try:
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, config, device, tokenizer
        )
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =============================================================================
    # LOAD BEST MODEL AND EVALUATE
    # =============================================================================
    print("\n🏆 Loading best model for final evaluation...")
    
    try:
        # Load best model
        best_model_path = os.path.join(config.out_dir, 'best_model.pt')
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"✅ Loaded best model from epoch {checkpoint['epoch'] + 1}")
        print(f"   📊 Best validation loss: {checkpoint['val_loss']:.4f}")
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_loss = evaluate_loss(model, test_loader, device)
        print(f"🎯 Final Test Loss: {test_loss:.4f}")
        
    except Exception as e:
        print(f"❌ Failed to load best model: {e}")
        print("Using current model state for evaluation...")
    
    # =============================================================================
    # FINAL TEXT GENERATION SHOWCASE
    # =============================================================================
    print("\n🎨 Final text generation showcase...")
    print("="*60)
    
    showcase_prompts = [
        "The meaning of life is",
        "To be happy, one must",
        "Philosophy teaches us that",
        "A wise person always",
        "The greatest virtue is"
    ]
    
    for i, prompt in enumerate(showcase_prompts, 1):
        print(f"\n🎭 Generation {i}:")
        print(f"📝 Prompt: '{prompt}'")
        
        try:
            generated_text = generate_text(
                model, tokenizer, device, prompt, 
                max_new_tokens=50, temperature=0.8, top_k=50
            )
            print(f"🤖 Generated: {generated_text}")
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    print("\n" + "="*60)
    print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)


# Run the main function
if __name__ == "__main__":
    main()