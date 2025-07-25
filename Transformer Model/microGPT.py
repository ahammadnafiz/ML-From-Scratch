import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken  
import os        
import urllib.request  
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# DATA PREPARATION AND TOKENIZATION
# =============================================================================

class TextDatasetForTraining(Dataset):
    """
    Creates a dataset from text for GPT training using a sliding window approach.
    
    This dataset takes a large text and creates many training examples by:
    1. Tokenizing the entire text
    2. Creating overlapping sequences of fixed length
    3. Each sequence becomes an input, with the target being the same sequence shifted by 1 position
    
    Example:
    If text is "Hello world how are you" and max_length=3, stride=2:
    - Input: [Hello, world, how] → Target: [world, how, are]
    - Input: [how, are, you] → Target: [are, you, <next_token>]
    """
    
    def __init__(self, text_content, tokenizer, sequence_length, sliding_window_stride):
        self.input_sequences = []  # Store input token sequences
        self.target_sequences = []  # Store target token sequences (shifted by 1)
        
        # Convert entire text to token IDs using the tokenizer
        all_token_ids = tokenizer.encode(text_content)
        
        # Create training examples using sliding window
        # We slide through the text creating overlapping sequences
        for start_position in range(0, len(all_token_ids) - sequence_length, sliding_window_stride):
            # Extract input sequence of desired length
            input_tokens = all_token_ids[start_position : start_position + sequence_length]
            # Target is the same sequence shifted right by 1 position (next token prediction)
            target_tokens = all_token_ids[start_position + 1 : start_position + sequence_length + 1]
            
            # Convert to tensors and store
            self.input_sequences.append(torch.tensor(input_tokens))
            self.target_sequences.append(torch.tensor(target_tokens))
            
    def __len__(self):
        """Return total number of training examples created"""
        return len(self.input_sequences)

    def __getitem__(self, index):
        """Get a single training example (input sequence, target sequence)"""
        return self.input_sequences[index], self.target_sequences[index]


# =============================================================================
# ATTENTION MECHANISM - THE HEART OF TRANSFORMERS
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism - the core innovation of transformers.
    
    This allows each token to "attend" to (look at) other tokens in the sequence
    to understand context. For example, in "The cat sat on the mat", when processing
    "sat", the attention can focus on "cat" to understand what is sitting.
    
    Key concepts:
    - Query (Q): What am I looking for?
    - Key (K): What information do I have?  
    - Value (V): The actual information content
    - Multiple heads: Look at different types of relationships simultaneously
    """
    
    def __init__(self, input_dimension, output_dimension, context_window_size, 
                 dropout_probability, number_of_attention_heads, use_bias_in_projections=False):
        super().__init__()
        
        # Ensure output dimension can be evenly split across attention heads
        assert output_dimension % number_of_attention_heads == 0, \
            f"Output dimension {output_dimension} must be divisible by number of heads {number_of_attention_heads}"
        
        self.output_dimension = output_dimension
        self.number_of_attention_heads = number_of_attention_heads
        self.dimension_per_attention_head = output_dimension // number_of_attention_heads
        
        # Pre-compute scaling factor for numerical stability
        # Larger dimensions need smaller attention scores to prevent extremely large values
        self.attention_scaling_factor = 1.0 / math.sqrt(self.dimension_per_attention_head)
        
        # Single linear layer that creates Query, Key, and Value projections simultaneously
        # This is more efficient than having 3 separate linear layers
        self.query_key_value_projection = nn.Linear(input_dimension, 3 * output_dimension, bias=use_bias_in_projections)
        
        # Final output projection to combine information from all attention heads
        self.output_projection = nn.Linear(output_dimension, output_dimension)
        self.attention_dropout = nn.Dropout(dropout_probability)
        
        # Create causal mask to prevent tokens from seeing future tokens
        # This mask ensures that token at position i can only attend to positions <= i
        # Upper triangular matrix of True values that will mask future positions
        self.register_buffer(
            "causal_attention_mask",
            torch.triu(torch.ones(context_window_size, context_window_size, dtype=torch.bool), diagonal=1)
        )

    def forward(self, input_embeddings):
        batch_size, sequence_length, embedding_dimension = input_embeddings.shape
        
        # Generate Query, Key, Value projections in one operation (more efficient)
        query_key_value_combined = self.query_key_value_projection(input_embeddings)  
        # Shape: (batch_size, sequence_length, 3 * output_dimension)
        
        # Reshape to separate Q, K, V and organize by attention heads
        # We want shape: (3, batch_size, num_heads, sequence_length, head_dimension)
        query_key_value_reshaped = query_key_value_combined.view(
            batch_size, sequence_length, 3, self.number_of_attention_heads, self.dimension_per_attention_head
        )
        query_key_value_rearranged = query_key_value_reshaped.permute(2, 0, 3, 1, 4)
        
        # Split into separate Query, Key, Value tensors
        queries, keys, values = query_key_value_rearranged.unbind(0)
        
        # Compute attention scores: how much should each token attend to every other token?
        # This is the core of the attention mechanism
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.attention_scaling_factor
        
        # Apply causal masking: prevent tokens from attending to future tokens
        # In language modeling, a token shouldn't know what comes next during training
        if sequence_length > 1:  # Skip masking for single tokens (efficiency)
            current_mask = self.causal_attention_mask[:sequence_length, :sequence_length]
            attention_scores = attention_scores.masked_fill(current_mask, float('-inf'))
        
        # Convert scores to probabilities (softmax) and apply dropout
        attention_probabilities = F.softmax(attention_scores, dim=-1)
        attention_probabilities = self.attention_dropout(attention_probabilities)
        
        # Apply attention probabilities to values: weighted combination of all values
        attention_output = torch.matmul(attention_probabilities, values)
        
        # Reshape back to combine all attention heads
        # Shape: (batch_size, sequence_length, output_dimension)
        combined_attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.output_dimension
        )
        
        # Final projection to combine information from all heads
        return self.output_projection(combined_attention_output)


# =============================================================================
# NORMALIZATION AND ACTIVATION FUNCTIONS
# =============================================================================

class LayerNormalization(nn.Module):
    """
    Layer Normalization: Normalizes inputs to have mean=0 and std=1.
    
    This helps with training stability by ensuring that values don't become
    too large or too small as they pass through many layers.
    
    Unlike batch normalization, layer norm works across the feature dimension
    for each example independently, making it suitable for sequential data.
    """
    
    def __init__(self, embedding_dimension):
        super().__init__()
        self.epsilon = 1e-5  # Small value to prevent division by zero
        
        # Learnable parameters to scale and shift the normalized values
        self.scale_parameter = nn.Parameter(torch.ones(embedding_dimension))
        self.shift_parameter = nn.Parameter(torch.zeros(embedding_dimension))

    def forward(self, input_tensor):
        # Calculate mean and variance across the last dimension (features)
        mean = input_tensor.mean(dim=-1, keepdim=True)
        variance = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize: subtract mean and divide by standard deviation
        normalized_input = (input_tensor - mean) / torch.sqrt(variance + self.epsilon)
        
        # Apply learnable scale and shift
        return self.scale_parameter * normalized_input + self.shift_parameter


class GELUActivation(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) activation function.
    
    GELU is a smooth activation function that works better than ReLU
    for transformer models. It's approximately: x * sigmoid(1.702 * x)
    
    This implementation uses the exact mathematical definition rather
    than approximations for better accuracy.
    """
    
    def forward(self, input_tensor):
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        return 0.5 * input_tensor * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2 / torch.pi)) * 
            (input_tensor + 0.044715 * torch.pow(input_tensor, 3))
        ))


# =============================================================================
# FEED-FORWARD NETWORK
# =============================================================================

class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    After attention, each token's representation is passed through this
    network independently. This adds non-linearity and allows the model
    to process the information gathered from attention.
    
    Standard transformer architecture: 
    - Expand dimension by 4x
    - Apply activation
    - Project back to original dimension
    """
    
    def __init__(self, model_configuration):
        super().__init__()
        embedding_dimension = model_configuration['embedding_dimension']
        
        # Two-layer network with expansion and contraction
        self.feed_forward_layers = nn.Sequential(
            # Expand to 4x the embedding dimension (standard transformer ratio)
            nn.Linear(embedding_dimension, 4 * embedding_dimension),
            GELUActivation(),  # Non-linear activation
            # Contract back to original embedding dimension
            nn.Linear(4 * embedding_dimension, embedding_dimension)
        )
    
    def forward(self, input_tensor):
        return self.feed_forward_layers(input_tensor)


# =============================================================================
# TRANSFORMER BLOCK - COMBINING ATTENTION AND FEED-FORWARD
# =============================================================================

class TransformerBlock(nn.Module):
    """
    A single Transformer block combining self-attention and feed-forward processing.
    
    This is the basic building block of GPT. Each block:
    1. Applies self-attention to let tokens communicate
    2. Applies feed-forward processing to each token independently
    3. Uses residual connections and layer normalization for stable training
    
    The residual connections help gradients flow during backpropagation,
    enabling training of very deep networks.
    """
    
    def __init__(self, model_configuration):
        super().__init__()
        
        # Self-attention mechanism
        self.self_attention = MultiHeadSelfAttention(
            input_dimension=model_configuration['embedding_dimension'],
            output_dimension=model_configuration['embedding_dimension'],
            context_window_size=model_configuration['context_window_size'],
            number_of_attention_heads=model_configuration['number_of_attention_heads'],
            dropout_probability=model_configuration['dropout_probability'],
            use_bias_in_projections=model_configuration['use_bias_in_projections']
        )
        
        # Feed-forward network
        self.feed_forward_network = FeedForwardNetwork(model_configuration)
        
        # Layer normalization (applied before attention and feed-forward, not after)
        self.layer_norm_before_attention = LayerNormalization(model_configuration['embedding_dimension'])
        self.layer_norm_before_feedforward = LayerNormalization(model_configuration['embedding_dimension'])
        
        # Dropout for residual connections
        self.residual_dropout = nn.Dropout(model_configuration['dropout_probability'])

    def forward(self, input_tensor):
        # Self-attention with residual connection (Pre-LayerNorm architecture)
        # This means we normalize BEFORE the sub-layer, not after
        residual_connection = input_tensor
        normalized_input = self.layer_norm_before_attention(input_tensor)
        attention_output = self.self_attention(normalized_input)
        attention_output = self.residual_dropout(attention_output)
        # Add residual connection: output = input + attention(norm(input))
        output_after_attention = attention_output + residual_connection
        
        # Feed-forward with residual connection
        residual_connection = output_after_attention
        normalized_input = self.layer_norm_before_feedforward(output_after_attention)
        feedforward_output = self.feed_forward_network(normalized_input)
        feedforward_output = self.residual_dropout(feedforward_output)
        # Add residual connection: output = input + feedforward(norm(input))
        final_output = feedforward_output + residual_connection
        
        return final_output


# =============================================================================
# MAIN GPT MODEL
# =============================================================================

class GPTLanguageModel(nn.Module):
    """
    Complete GPT (Generative Pre-trained Transformer) model.
    
    This model generates text by predicting the next token in a sequence.
    It uses the transformer architecture with:
    - Token embeddings: Convert tokens to dense vectors
    - Positional embeddings: Add position information
    - Multiple transformer blocks: Process and refine representations
    - Output head: Convert final representations back to token probabilities
    """
    
    def __init__(self, model_configuration):
        super().__init__()
        
        # Token embeddings: convert token IDs to dense vectors
        # Each token in the vocabulary gets its own learnable vector representation
        self.token_embedding_layer = nn.Embedding(
            model_configuration['vocabulary_size'], 
            model_configuration['embedding_dimension']
        )
        
        # Positional embeddings: add information about token positions
        # Since attention doesn't inherently understand order, we add position info
        self.positional_embedding_layer = nn.Embedding(
            model_configuration['context_window_size'], 
            model_configuration['embedding_dimension']
        )
        
        # Dropout for embeddings (regularization)
        self.embedding_dropout = nn.Dropout(model_configuration['dropout_probability'])
        
        # Stack of transformer blocks - this is where the magic happens
        # Each block allows tokens to communicate and process information
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(model_configuration) for _ in range(model_configuration['number_of_layers'])]
        )
        
        # Final layer normalization before output
        self.final_layer_normalization = LayerNormalization(model_configuration['embedding_dimension'])
        
        # Output head: convert embeddings back to vocabulary probabilities
        # No bias needed here - it's a common practice in language models
        self.language_model_head = nn.Linear(
            model_configuration['embedding_dimension'], 
            model_configuration['vocabulary_size'], 
            bias=False
        )

    def forward(self, input_token_ids):
        batch_size, sequence_length = input_token_ids.shape
        
        # Get token embeddings for each input token
        token_embeddings = self.token_embedding_layer(input_token_ids)
        
        # Get positional embeddings for each position in the sequence
        position_indices = torch.arange(sequence_length, device=input_token_ids.device)
        positional_embeddings = self.positional_embedding_layer(position_indices)
        
        # Combine token and positional embeddings
        # This gives each token both semantic meaning (from token embedding)
        # and positional information (from positional embedding)
        combined_embeddings = token_embeddings + positional_embeddings
        embedded_input = self.embedding_dropout(combined_embeddings)
        
        # Pass through all transformer blocks sequentially
        # Each block refines the representations by allowing tokens to communicate
        processed_representations = self.transformer_blocks(embedded_input)
        
        # Apply final layer normalization
        normalized_output = self.final_layer_normalization(processed_representations)
        
        # Convert to vocabulary probabilities
        # Shape: (batch_size, sequence_length, vocabulary_size)
        output_logits = self.language_model_head(normalized_output)
        
        return output_logits


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Configuration matching GPT-2 Small (124M parameters)
GPT2_SMALL_CONFIGURATION = {
    "vocabulary_size": 50257,        # Size of GPT-2 vocabulary
    "context_window_size": 1024,     # Maximum sequence length
    "embedding_dimension": 768,      # Size of token embeddings
    "number_of_attention_heads": 12, # Number of attention heads per block
    "number_of_layers": 12,          # Number of transformer blocks
    "dropout_probability": 0.1,      # Dropout rate for regularization
    "use_bias_in_projections": False # Whether to use bias in QKV projections
}


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def create_training_dataloader(text_content, batch_size=4, sequence_length=256,
                               sliding_window_stride=128, shuffle_data=True, 
                               drop_incomplete_batches=True, number_of_workers=0):
    """
    Creates a PyTorch DataLoader for training the GPT model.
    
    Args:
        text_content: Raw text string to train on
        batch_size: Number of sequences per batch
        sequence_length: Length of each sequence
        sliding_window_stride: How much to shift window between sequences
        shuffle_data: Whether to shuffle training examples
        drop_incomplete_batches: Whether to drop the last incomplete batch
        number_of_workers: Number of workers for data loading
    
    Returns:
        DataLoader ready for training
    """
    # Initialize the GPT-2 tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    
    # Create dataset from text
    dataset = TextDatasetForTraining(
        text_content=text_content,
        tokenizer=tokenizer, 
        sequence_length=sequence_length,
        sliding_window_stride=sliding_window_stride
    )
    
    # Create and return dataloader
    dataloader = DataLoader(
         dataset,
         batch_size=batch_size,
         shuffle=shuffle_data,
         drop_last=drop_incomplete_batches, 
         num_workers=number_of_workers 
    )
    return dataloader


# =============================================================================
# TEXT GENERATION FUNCTIONS
# =============================================================================

def generate_text_greedy(model, starting_token_ids, maximum_new_tokens, context_window_size):
    """
    Generate text using greedy decoding (always pick the most likely next token).
    
    This is deterministic - given the same input, it will always produce the same output.
    Good for consistency but can be repetitive.
    
    Args:
        model: The trained GPT model
        starting_token_ids: Initial tokens to start generation from
        maximum_new_tokens: How many new tokens to generate
        context_window_size: Maximum context window the model can handle
    
    Returns:
        Token IDs including the starting tokens plus generated tokens
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    
    for step in range(maximum_new_tokens):
        # Only use the last 'context_window_size' tokens if sequence is too long
        # This prevents exceeding the model's maximum context length
        current_context = starting_token_ids[:, -context_window_size:]
        
        # Generate predictions without computing gradients (faster)
        with torch.no_grad():
            output_logits = model(current_context)
        
        # Get logits for the last token position (what comes next?)
        next_token_logits = output_logits[:, -1, :]
        
        # Pick the token with highest probability (greedy)
        next_token_probabilities = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.argmax(next_token_probabilities, dim=-1, keepdim=True)
        
        # Add the predicted token to our sequence
        starting_token_ids = torch.cat((starting_token_ids, next_token_id), dim=1)
    
    return starting_token_ids


def generate_text_with_temperature(model, starting_token_ids, maximum_new_tokens, 
                                   context_window_size, temperature=1.0, top_k_tokens=None):
    """
    Generate text with temperature sampling for more diverse outputs.
    
    Temperature controls randomness:
    - temperature=0.0: Deterministic (same as greedy)
    - temperature=1.0: Sample according to model's probabilities  
    - temperature>1.0: More random, less coherent
    - temperature<1.0: Less random, more focused
    
    Top-k sampling only considers the k most likely tokens at each step.
    
    Args:
        model: The trained GPT model
        starting_token_ids: Initial tokens to start generation from
        maximum_new_tokens: How many new tokens to generate
        context_window_size: Maximum context window the model can handle
        temperature: Controls randomness of generation
        top_k_tokens: If specified, only consider top k tokens at each step
    
    Returns:
        Token IDs including the starting tokens plus generated tokens
    """
    model.eval()
    
    for step in range(maximum_new_tokens):
        # Limit context to model's maximum window size
        current_context = starting_token_ids[:, -context_window_size:]
        
        with torch.no_grad():
            output_logits = model(current_context)
        
        # Get logits for next token prediction
        next_token_logits = output_logits[:, -1, :]

        # Apply top-k filtering if specified
        if top_k_tokens is not None:
            # Keep only the top k most likely tokens
            top_k_values, top_k_indices = torch.topk(next_token_logits, top_k_tokens)
            # Set all other tokens to negative infinity (zero probability)
            min_top_k_value = top_k_values[:, -1]
            next_token_logits = torch.where(
                next_token_logits < min_top_k_value, 
                torch.tensor(float("-inf")).to(next_token_logits.device), 
                next_token_logits
            )

        # Apply temperature scaling
        if temperature > 0.0:
            # Scale logits by temperature (higher temp = more random)
            scaled_logits = next_token_logits / temperature
            # Convert to probabilities
            next_token_probabilities = torch.softmax(scaled_logits, dim=-1)
            # Sample from the probability distribution
            next_token_id = torch.multinomial(next_token_probabilities, num_samples=1)
        else:
            # Temperature = 0 means greedy decoding
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Add predicted token to sequence
        starting_token_ids = torch.cat((starting_token_ids, next_token_id), dim=1)

    return starting_token_ids


# =============================================================================
# TRAINING LOSS CALCULATION
# =============================================================================

def calculate_loss_for_single_batch(input_batch, target_batch, model, device):
    """
    Calculate cross-entropy loss for a single batch.
    
    Cross-entropy loss measures how far off the model's predictions are
    from the actual next tokens. Lower loss = better predictions.
    
    Args:
        input_batch: Input token sequences
        target_batch: Target token sequences (input shifted by 1)
        model: The GPT model
        device: Device to run computations on (CPU/GPU)
    
    Returns:
        Average loss across all tokens in the batch
    """
    # Move data to the specified device (GPU/CPU)
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    # Get model predictions
    prediction_logits = model(input_batch)
    
    # Reshape for cross-entropy loss calculation
    # From (batch_size, sequence_length, vocabulary_size) to (batch_size * sequence_length, vocabulary_size)
    flattened_predictions = prediction_logits.flatten(0, 1)
    flattened_targets = target_batch.flatten()
    
    # Calculate cross-entropy loss
    loss = torch.nn.functional.cross_entropy(flattened_predictions, flattened_targets)
    return loss


def calculate_average_loss_over_dataloader(dataloader, model, device, max_batches_to_evaluate=None):
    """
    Calculate average loss over an entire dataset.
    
    This is used to evaluate model performance on training/validation sets.
    
    Args:
        dataloader: DataLoader containing the dataset
        model: The GPT model
        device: Device for computations
        max_batches_to_evaluate: Limit evaluation to this many batches (for speed)
    
    Returns:
        Average loss across all evaluated batches
    """
    total_loss = 0.0
    
    # Handle empty dataloader
    if len(dataloader) == 0:
        return float("nan")
    
    # Determine how many batches to evaluate
    if max_batches_to_evaluate is None:
        batches_to_process = len(dataloader)
    else:
        batches_to_process = min(max_batches_to_evaluate, len(dataloader))
    
    # Calculate loss for each batch
    for batch_index, (input_batch, target_batch) in enumerate(dataloader):
        if batch_index < batches_to_process:
            batch_loss = calculate_loss_for_single_batch(input_batch, target_batch, model, device)
            total_loss += batch_loss.item()
        else:
            break
            
    return total_loss / batches_to_process


def evaluate_model_performance(model, training_dataloader, validation_dataloader, 
                              device, max_evaluation_batches):
    """
    Evaluate model on both training and validation sets.
    
    This helps us understand:
    - How well the model fits the training data
    - Whether the model generalizes to unseen data (validation)
    - If we're overfitting (training loss much lower than validation loss)
    
    Args:
        model: The GPT model to evaluate
        training_dataloader: DataLoader for training set
        validation_dataloader: DataLoader for validation set  
        device: Device for computations
        max_evaluation_batches: Limit evaluation to this many batches per set
    
    Returns:
        Tuple of (training_loss, validation_loss)
    """
    model.eval()  # Set to evaluation mode
    
    with torch.no_grad():  # Don't compute gradients during evaluation
        training_loss = calculate_average_loss_over_dataloader(
            training_dataloader, model, device, max_evaluation_batches
        )
        validation_loss = calculate_average_loss_over_dataloader(
            validation_dataloader, model, device, max_evaluation_batches
        )
    
    model.train()  # Set back to training mode
    return training_loss, validation_loss


# =============================================================================
# TEXT PROCESSING UTILITIES
# =============================================================================

def convert_text_to_token_ids(text, tokenizer):
    """Convert text string to tensor of token IDs"""
    encoded_tokens = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # Add batch dimension (unsqueeze(0))
    token_tensor = torch.tensor(encoded_tokens).unsqueeze(0)
    return token_tensor


def convert_token_ids_to_text(token_ids, tokenizer):
    """Convert tensor of token IDs back to text string"""
    # Remove batch dimension (squeeze(0))
    flattened_tokens = token_ids.squeeze(0)
    decoded_text = tokenizer.decode(flattened_tokens.tolist())
    return decoded_text


def generate_and_display_sample_text(model, tokenizer, device, starting_text):
    """
    Generate and print a sample text starting from given prompt.
    
    This is useful for monitoring training progress - we can see how
    the model's text generation improves over time.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer for text conversion
        device: Device for computations
        starting_text: Text prompt to start generation from
    """
    model.eval()
    
    # Get the maximum context window size from the model
    max_context_size = model.positional_embedding_layer.weight.shape[0]
    
    # Convert starting text to token IDs
    starting_tokens = convert_text_to_token_ids(starting_text, tokenizer).to(device)
    
    # Generate text without computing gradients (faster)
    with torch.no_grad():
        generated_tokens = generate_text_greedy(
            model=model, 
            starting_token_ids=starting_tokens,
            maximum_new_tokens=50,  # Generate 50 new tokens
            context_window_size=max_context_size
        )
    
    # Convert back to text and display
    generated_text = convert_token_ids_to_text(generated_tokens, tokenizer)
    # Replace newlines with spaces for compact display
    print(generated_text.replace("\n", " "))
    
    model.train()  # Set back to training mode


# =============================================================================
# COMPLETE TRAINING PIPELINE
# =============================================================================

def train_gpt_model_complete_pipeline(model, training_dataloader, validation_dataloader, 
                                     optimizer, device, number_of_epochs,
                                     evaluation_frequency, max_evaluation_batches, 
                                     sample_generation_prompt, tokenizer):
    """
    Complete training loop with progress monitoring and text generation.
    
    This function:
    1. Trains the model for specified number of epochs
    2. Periodically evaluates on training/validation sets
    3. Generates sample text to monitor progress
    4. Tracks training statistics
    
    Args:
        model: GPT model to train
        training_dataloader: DataLoader for training data
        validation_dataloader: DataLoader for validation data
        optimizer: Optimizer for updating model weights
        device: Device for computations (CPU/GPU)
        number_of_epochs: How many times to go through the training data
        evaluation_frequency: How often to evaluate (every N steps)
        max_evaluation_batches: Limit evaluation to N batches for speed
        sample_generation_prompt: Text prompt for monitoring generation quality
        tokenizer: Tokenizer for text conversion
    
    Returns:
        Dictionary containing training statistics and final model
    """
    # Initialize tracking variables
    training_losses = []
    validation_losses = []
    tokens_processed_timeline = []
    total_tokens_processed = 0
    global_training_step = -1

    print("Starting training...")
    print("=" * 60)

    # Main training loop
    for current_epoch in range(number_of_epochs):
        model.train()  # Ensure model is in training mode
        
        print(f"\nEpoch {current_epoch + 1}/{number_of_epochs}")
        print("-" * 40)
        
        # Process each batch in the training dataloader
        for batch_index, (input_batch, target_batch) in enumerate(training_dataloader):
            global_training_step += 1
            
            # Calculate and accumulate tokens processed
            batch_size, sequence_length = input_batch.shape
            tokens_in_batch = batch_size * sequence_length
            total_tokens_processed += tokens_in_batch
            
            # Forward pass: compute predictions and loss
            optimizer.zero_grad()  # Clear previous gradients
            batch_loss = calculate_loss_for_single_batch(input_batch, target_batch, model, device)
            
            # Backward pass: compute gradients and update weights
            batch_loss.backward()
            optimizer.step()
            
            # Periodically evaluate and generate sample text
            if global_training_step % evaluation_frequency == 0:
                print(f"\nStep {global_training_step} | Tokens processed: {total_tokens_processed:,}")
                
                # Evaluate on training and validation sets
                train_loss, val_loss = evaluate_model_performance(
                    model, training_dataloader, validation_dataloader, 
                    device, max_evaluation_batches
                )
                
                # Store evaluation results
                training_losses.append(train_loss)
                validation_losses.append(val_loss)
                tokens_processed_timeline.append(total_tokens_processed)
                
                print(f"Training loss: {train_loss:.4f} | Validation loss: {val_loss:.4f}")
                
                # Generate and display sample text
                print("Sample generation:")
                generate_and_display_sample_text(model, tokenizer, device, sample_generation_prompt)
                print()
        
        print(f"Completed epoch {current_epoch + 1}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    
    # Return training results
    return {
        'model': model,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'tokens_processed': tokens_processed_timeline,
        'total_tokens_processed': total_tokens_processed,
        'final_training_loss': training_losses[-1] if training_losses else None,
        'final_validation_loss': validation_losses[-1] if validation_losses else None
    }


# =============================================================================
# MODEL INITIALIZATION AND UTILITIES
# =============================================================================

def initialize_gpt_model_weights(model):
    """
    Initialize model weights using best practices for transformer training.
    
    This initialization scheme helps with:
    - Faster convergence during training
    - Better gradient flow
    - Avoiding vanishing/exploding gradients
    
    Args:
        model: GPT model to initialize
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization for linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal initialization for embedding layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def count_model_parameters(model):
    """
    Count the total number of trainable parameters in the model.
    
    This helps understand model size and computational requirements.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_gpt_model(config=None, device='cpu'):
    """
    Factory function to create and initialize a GPT model.
    
    Args:
        config: Model configuration dictionary (uses GPT2_SMALL_CONFIGURATION if None)
        device: Device to move model to
        
    Returns:
        Initialized GPT model ready for training
    """
    if config is None:
        config = GPT2_SMALL_CONFIGURATION
    
    # Create model
    model = GPTLanguageModel(config)
    
    # Initialize weights
    initialize_gpt_model_weights(model)
    
    # Move to device
    model = model.to(device)
    
    # Print model information
    num_params = count_model_parameters(model)
    print(f"Created GPT model with {num_params:,} parameters")
    print(f"Model configuration: {config}")
    
    return model


def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint for later resuming training.
    
    Args:
        model: Trained model
        optimizer: Optimizer state
        epoch: Current epoch number
        loss: Current loss value
        filepath: Where to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")


def load_model_checkpoint(model, optimizer, filepath, device):
    """
    Load model checkpoint to resume training.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (epoch, loss) from checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Loaded checkpoint from {filepath}")
    print(f"Resuming from epoch {epoch}, loss: {loss:.4f}")
    
    return epoch, loss


# =============================================================================
# DEMONSTRATION AND TESTING
# =============================================================================

def download_and_load_text(url, filename):
    """
    Download text file from URL if it doesn't exist locally, then load and return its content.
    
    This function helps us work with remote text datasets without manually downloading them.
    It implements a simple caching mechanism - if the file already exists locally,
    it skips the download step and just loads the existing file.
    
    Args:
        url (str): The URL where the text file is hosted
        filename (str): Local filename to save the downloaded file as
        
    Returns:
        str: The complete text content of the file
        
    Raises:
        Various exceptions related to network issues, file permissions, etc.
    """
    # Check if the file already exists locally to avoid unnecessary downloads
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        # Download the file from the URL and save it locally
        # urllib.request.urlretrieve() downloads the file and saves it to disk
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"File {filename} already exists, using cached version")
    
    # Open and read the text file with UTF-8 encoding (handles most text files)
    # Using 'with' statement ensures the file is properly closed after reading
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def demo_microGPT():
    """
    Demonstration function showing how to use the microGPT implementation.
    
    This creates a small model, trains it on sample text, and shows generation.
    """
    print("=" * 80)
    print("MICROGPT DEMONSTRATION")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a smaller configuration for demo
    demo_config = {
        "vocabulary_size": 50257,
        "context_window_size": 256,      # Smaller context for demo
        "embedding_dimension": 384,      # Smaller embedding dimension
        "number_of_attention_heads": 6,  # Fewer attention heads
        "number_of_layers": 6,           # Fewer layers
        "dropout_probability": 0.1,
        "use_bias_in_projections": False
    }
    
    # Load text data from local file
    print("Loading text data from local file...")
    
    # Specify the path to your text file
    text_file_path = "training_data.txt"  # Change this to your actual file path
    
    try:
        # Load text from local file
        with open(text_file_path, 'r', encoding='utf-8') as file:
            sample_text = file.read()
        
        print(f"Successfully loaded text from '{text_file_path}'")
        print(f"File contains {len(sample_text):,} characters")
        
        # Check if file has sufficient content
        if len(sample_text) < 1000:
            print("Warning: Text file is quite small. Consider using a larger text file for better training results.")
        
    except FileNotFoundError:
        print(f"Error: File '{text_file_path}' not found!")
        print("Creating fallback text data...")
        
        # Fallback to sample text if file not found
        sample_text = """Every effort moves you forward. Every step brings you closer to your goal. 
        The journey of learning is continuous. Knowledge grows with every experience. 
        Progress happens when we persist through challenges. Success comes to those who never give up.
        The meaning of life is to find purpose and happiness. In the future, technology will transform our world.
        Every effort moves you closer to your goals. The meaning of life is discovered through experiences and relationships.
        In the future, we will solve many challenges. Every effort moves you toward success.
        The meaning of life is to grow and learn continuously. In the future, artificial intelligence will help humanity.
        Every effort moves you beyond your limits. The meaning of life is to make a positive impact.
        In the future, we will explore new frontiers. Every effort moves you forward despite obstacles.
        The meaning of life is found in love and connection. In the future, sustainable solutions will preserve our planet.
        Every effort moves you to new heights. The meaning of life is to pursue your passions. 
        In the future, education will be more accessible. Every effort moves you closer to understanding.
        The meaning of life is to help others succeed. In the future, collaboration will solve global problems.
        """ * 20  # Repeat to create sufficient training data
        
        print("Using fallback text data for training")
    
    except Exception as e:
        print(f"Error reading file '{text_file_path}': {e}")
        print("Creating fallback text data...")
        
        # Fallback text in case of any other error
        sample_text = """Every effort moves you forward. Every step brings you closer to your goal. 
        The journey of learning is continuous. Knowledge grows with every experience. 
        Progress happens when we persist through challenges. Success comes to those who never give up.
        The meaning of life is to find purpose and happiness. In the future, technology will transform our world.
        """ * 50  # Repeat to create sufficient training data
        
        print("Using fallback text data for training")
    
    print(f"Final text length: {len(sample_text):,} characters")
    print("Text data ready for training")
    
    # Create model
    model = create_gpt_model(demo_config, device)
    
    # Create dataloaders
    print("\nCreating training dataloader...")
    train_dataloader = create_training_dataloader(
        text_content=sample_text,
        batch_size=4,
        sequence_length=64,
        sliding_window_stride=32,
        shuffle_data=True
    )
    
    # For demo, use same dataloader for validation (in practice, use different data)
    val_dataloader = train_dataloader
    
    print(f"Created dataloader with {len(train_dataloader)} batches")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    
    print("\nStarting training...")
    
    # Train the model
    training_results = train_gpt_model_complete_pipeline(
        model=model,
        training_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        number_of_epochs=20,          # Increased to 20 epochs for better training
        evaluation_frequency=10,       # Evaluate every 10 steps
        max_evaluation_batches=5,     # Evaluate on 5 batches max
        sample_generation_prompt="Every effort moves you",
        tokenizer=tokenizer
    )
    
    print("\nTraining completed!")
    print(f"Final training loss: {training_results['final_training_loss']:.4f}")
    print(f"Final validation loss: {training_results['final_validation_loss']:.4f}")
    
    # Demonstrate text generation with different methods
    print("\n" + "=" * 50)
    print("TEXT GENERATION COMPARISON - DIFFERENT METHODS")
    print("=" * 50)
    
    model.eval()
    generation_prompts = [
        "Every effort moves you",
        "The meaning of life is",
        "In the future"
    ]
    
    for prompt in generation_prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: '{prompt}'")
        print('='*60)
        
        # Convert prompt to tokens once for all methods
        prompt_tokens = convert_text_to_token_ids(prompt, tokenizer).to(device)
        
        # Method 1: Greedy Decoding (Deterministic)
        print(f"\n1. GREEDY DECODING (Deterministic):")
        print("-" * 40)
        with torch.no_grad():
            greedy_tokens = generate_text_greedy(
                model=model,
                starting_token_ids=prompt_tokens.clone(),
                maximum_new_tokens=25,
                context_window_size=demo_config["context_window_size"]
            )
        greedy_text = convert_token_ids_to_text(greedy_tokens, tokenizer)
        print(f"Output: {greedy_text}")
        
        # Method 2: Temperature Sampling (Low Temperature - More Focused)
        print(f"\n2. TEMPERATURE SAMPLING (Low Temp = 0.3 - Focused):")
        print("-" * 40)
        with torch.no_grad():
            low_temp_tokens = generate_text_with_temperature(
                model=model,
                starting_token_ids=prompt_tokens.clone(),
                maximum_new_tokens=25,
                context_window_size=demo_config["context_window_size"],
                temperature=0.3  # Low temperature for focused output
            )
        low_temp_text = convert_token_ids_to_text(low_temp_tokens, tokenizer)
        print(f"Output: {low_temp_text}")
        
        # Method 3: Temperature Sampling (Medium Temperature - Balanced)
        print(f"\n3. TEMPERATURE SAMPLING (Medium Temp = 0.7 - Balanced):")
        print("-" * 40)
        with torch.no_grad():
            med_temp_tokens = generate_text_with_temperature(
                model=model,
                starting_token_ids=prompt_tokens.clone(),
                maximum_new_tokens=25,
                context_window_size=demo_config["context_window_size"],
                temperature=0.7  # Medium temperature for balanced creativity
            )
        med_temp_text = convert_token_ids_to_text(med_temp_tokens, tokenizer)
        print(f"Output: {med_temp_text}")
        
        # Method 4: Temperature Sampling (High Temperature - More Creative)
        print(f"\n4. TEMPERATURE SAMPLING (High Temp = 1.2 - Creative):")
        print("-" * 40)
        with torch.no_grad():
            high_temp_tokens = generate_text_with_temperature(
                model=model,
                starting_token_ids=prompt_tokens.clone(),
                maximum_new_tokens=25,
                context_window_size=demo_config["context_window_size"],
                temperature=1.2  # High temperature for creative output
            )
        high_temp_text = convert_token_ids_to_text(high_temp_tokens, tokenizer)
        print(f"Output: {high_temp_text}")
        
        # Method 5: Top-K Sampling with Temperature (Most Sophisticated)
        print(f"\n5. TOP-K + TEMPERATURE (K=20, Temp=0.8 - Sophisticated):")
        print("-" * 40)
        with torch.no_grad():
            topk_tokens = generate_text_with_temperature(
                model=model,
                starting_token_ids=prompt_tokens.clone(),
                maximum_new_tokens=25,
                context_window_size=demo_config["context_window_size"],
                temperature=0.8,  # Balanced temperature
                top_k_tokens=20   # Only consider top 20 most likely tokens
            )
        topk_text = convert_token_ids_to_text(topk_tokens, tokenizer)
        print(f"Output: {topk_text}")
        
        # Method 6: Top-K Sampling with Lower Temperature (Conservative but diverse)
        print(f"\n6. TOP-K + LOW TEMPERATURE (K=10, Temp=0.5 - Conservative):")
        print("-" * 40)
        with torch.no_grad():
            conservative_tokens = generate_text_with_temperature(
                model=model,
                starting_token_ids=prompt_tokens.clone(),
                maximum_new_tokens=25,
                context_window_size=demo_config["context_window_size"],
                temperature=0.5,  # Low temperature for focus
                top_k_tokens=10   # Only consider top 10 most likely tokens
            )
        conservative_text = convert_token_ids_to_text(conservative_tokens, tokenizer)
        print(f"Output: {conservative_text}")
        
        print(f"\n" + "="*60)
        print("GENERATION METHOD COMPARISON:")
        print("- Greedy: Always picks most likely token (deterministic)")
        print("- Low Temp (0.3): Focused, conservative, less creative")
        print("- Medium Temp (0.7): Balanced creativity and coherence")
        print("- High Temp (1.2): More creative but potentially less coherent")
        print("- Top-K + Temp: Combines vocabulary filtering with temperature")
        print("- Conservative: Safe choices with limited vocabulary")
        print("="*60)
    
    print("\n" + "=" * 80)
    print("MICROGPT COMPREHENSIVE DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print("Training Summary:")
    print(f"- Epochs: 20 (increased for better learning)")
    print(f"- Final Training Loss: {training_results['final_training_loss']:.4f}")
    print(f"- Final Validation Loss: {training_results['final_validation_loss']:.4f}")
    print(f"- Total Tokens Processed: {training_results['total_tokens_processed']:,}")
    print("\nGeneration Methods Tested:")
    print("1. Greedy Decoding (deterministic)")
    print("2. Low Temperature Sampling (focused)")
    print("3. Medium Temperature Sampling (balanced)")
    print("4. High Temperature Sampling (creative)")
    print("5. Top-K + Temperature (sophisticated)")
    print("6. Conservative Top-K (safe choices)")
    print("\nThis comprehensive test shows how different generation strategies")
    print("affect the creativity, coherence, and diversity of generated text!")
    print("=" * 80)
    
    return model, training_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the demonstration
    model, results = demo_microGPT()