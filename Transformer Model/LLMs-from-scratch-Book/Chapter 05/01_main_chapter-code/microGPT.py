import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Core Components
# =============================================================================

class GPTDataset(Dataset):
    """Dataset for GPT training with sliding window"""
    
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class MultiHeadAttention(nn.Module):
    """Optimized multi-head causal attention mechanism"""
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)  # Pre-compute scale factor
        
        # Single linear layer for QKV (3x more efficient than 3 separate layers)
        self.qkv_proj = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as buffer (avoid recreating each forward pass)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # Single QKV projection (more efficient than 3 separate projections)
        qkv = self.qkv_proj(x)  # (b, num_tokens, 3 * d_out)
        
        # Split and reshape in one operation
        qkv = qkv.view(b, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)
        
        # Scaled dot-product attention with fused operations
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        
        # Apply causal mask (use pre-computed boolean mask)
        if num_tokens > 1:  # Skip masking for single token
            attn_scores = attn_scores.masked_fill(
                self.mask[:num_tokens, :num_tokens], 
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context_vec = torch.matmul(attn_weights, values)
        
        # Reshape back to original format
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        
        # Output projection
        return self.out_proj(context_vec)


class LayerNorm(nn.Module):
    """Layer normalization"""
    
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
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Feed-forward network"""
    
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
    """Transformer block with attention and feed-forward"""
    
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
        # Multi-head attention with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        # Feed-forward with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x


class GPTModel(nn.Module):
    """Main GPT model"""
    
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

# GPT-2 124M configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


def create_dataloader(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
         dataset,
         batch_size=batch_size,
         shuffle=shuffle,
         drop_last=drop_last, 
         num_workers=num_workers 
         )
    return dataloader


def generate_text(model, idx, max_new_tokens, context_size, temperature=1.0):
    """Generate text with temperature sampling"""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probas, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx


# =============================================================================
# Training and Evaluation Functions
# =============================================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a single batch"""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss over a data loader"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluate model on train and validation sets"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Simple text generation function for evaluation"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx


def text_to_token_ids(text, tokenizer):
    """Convert text to token IDs"""
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """Convert token IDs to text"""
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generate and print a sample text"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                      eval_freq, eval_iter, start_context, tokenizer):
    """Complete training loop with evaluation and text generation"""
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        print(f"\nEpoch {epoch+1} sample generation:")
        generate_and_print_sample(model, tokenizer, device, start_context)
        print()

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """Plot training and validation losses"""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, label="Validation loss", linestyle="--")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.set_title("Loss vs Epochs")
        ax1.grid(True)
        
        # Plot against tokens seen
        ax2.plot(tokens_seen, train_losses, label="Training loss")
        ax2.plot(tokens_seen, val_losses, label="Validation loss", linestyle="--")
        ax2.set_xlabel("Tokens seen")
        ax2.set_ylabel("Loss")
        ax2.legend(loc="upper right")
        ax2.set_title("Loss vs Tokens Seen")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")


def calculate_perplexity(loss):
    """Calculate perplexity from loss"""
    return torch.exp(torch.tensor(loss)).item()


def complete_training_pipeline(text_file_path, config=None, num_epochs=10, learning_rate=0.0004, 
                             batch_size=2, eval_freq=5, eval_iter=5, train_ratio=0.90):
    """Complete training pipeline from data loading to evaluation"""
    
    if config is None:
        config = GPT_CONFIG_124M.copy()
        config["context_length"] = 256  # Shorter for faster training
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading data...")
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text_data = f.read()
    except FileNotFoundError:
        print(f"Error: File {text_file_path} not found!")
        return None
    
    print(f"Text length: {len(text_data)} characters")
    
    # Split data
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_dataloader(
        train_data,
        batch_size=batch_size,
        max_length=config["context_length"],
        stride=config["context_length"],
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    val_loader = create_dataloader(
        val_data,
        batch_size=batch_size,
        max_length=config["context_length"],
        stride=config["context_length"],
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    torch.manual_seed(123)
    model = GPTModel(config)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    
    # Initial evaluation
    print("\nInitial evaluation:")
    tokenizer = tiktoken.get_encoding("gpt2")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print(f"Initial - Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
    print(f"Initial perplexity - Train: {calculate_perplexity(train_loss):.1f}, Val: {calculate_perplexity(val_loss):.1f}")
    
    # Training
    print(f"\nStarting training for {num_epochs} epochs...")
    start_context = "Every effort moves you"
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
        start_context=start_context, tokenizer=tokenizer
    )
    
    # Final evaluation
    print("\nFinal evaluation:")
    with torch.no_grad():
        final_train_loss = calc_loss_loader(train_loader, model, device)
        final_val_loss = calc_loss_loader(val_loader, model, device)
    print(f"Final - Train loss: {final_train_loss:.3f}, Val loss: {final_val_loss:.3f}")
    print(f"Final perplexity - Train: {calculate_perplexity(final_train_loss):.1f}, Val: {calculate_perplexity(final_val_loss):.1f}")
    
    # Plot results
    if train_losses:
        epochs_seen = []
        for i in range(len(train_losses)):
            step = (i + 1) * eval_freq
            epochs_seen.append(step / len(train_loader))
        
        plot_losses(epochs_seen, tokens_seen, train_losses, val_losses)
    
    # Generate final samples
    print("\nFinal text generation samples:")
    test_prompts = [
        "Every effort moves you",
        "The meaning of life is",
        "In a distant future",
        "The old man"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        generate_and_print_sample(model, tokenizer, device, prompt)
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'tokens_seen': tokens_seen,
        'config': config,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss
    }


# =============================================================================
# Additional Utility Functions
# =============================================================================

def save_model(model, filepath, metadata=None):
    """Save model with optional metadata"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, config):
    """Load model from file"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model = GPTModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    metadata = checkpoint.get('metadata', {})
    print(f"Model loaded from {filepath}")
    return model, metadata


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def generate_interactive(model, tokenizer, device, max_tokens=50):
    """Interactive text generation"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    
    print("Interactive Text Generation (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nEnter prompt: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
                
            encoded = text_to_token_ids(user_input, tokenizer).to(device)
            
            with torch.no_grad():
                token_ids = generate_text_simple(
                    model=model, idx=encoded,
                    max_new_tokens=max_tokens, context_size=context_size
                )
            
            decoded_text = token_ids_to_text(token_ids, tokenizer)
            print(f"Generated: {decoded_text}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")


# =============================================================================
# Example Usage and Main Functions
# =============================================================================

def main():
    """Example usage of the complete LLM training pipeline"""
    print("=" * 70)
    print("MicroGPT Complete Training Pipeline")
    print("=" * 70)
    
    # Run complete training pipeline
    text_file = 'the-verdict.txt'
    
    # You can customize these parameters
    training_config = {
        'num_epochs': 10,
        'learning_rate': 0.0004,
        'batch_size': 2,
        'eval_freq': 5,
        'eval_iter': 5,
        'train_ratio': 0.90
    }
    
    print(f"Training configuration: {training_config}")
    
    results = complete_training_pipeline(text_file, **training_config)
    
    if results:
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print("=" * 70)
        
        # Print summary
        print(f"Final training loss: {results['final_train_loss']:.3f}")
        print(f"Final validation loss: {results['final_val_loss']:.3f}")
        print(f"Training loss reduction: {results['train_losses'][0] - results['final_train_loss']:.3f}")
        print(f"Validation loss reduction: {results['val_losses'][0] - results['final_val_loss']:.3f}")
        
        return results
    else:
        print("Training failed!")
        return None


def demo_basic_usage():
    """Demo basic model usage without training"""
    print("\n" + "=" * 50)
    print("Basic Model Demo (No Training)")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    config = GPT_CONFIG_124M.copy()
    config["context_length"] = 256
    model = GPTModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Example text generation (untrained model)
    tokenizer = tiktoken.get_encoding('gpt2')
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    print(f"\nInput: '{start_context}'")
    
    # Generate text
    output = generate_text(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=config["context_length"],
        temperature=0.7
    )
    
    decoded_text = tokenizer.decode(output.squeeze(0).tolist())
    print(f"Generated (untrained): '{decoded_text}'")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_basic_usage()
    else:
        main()