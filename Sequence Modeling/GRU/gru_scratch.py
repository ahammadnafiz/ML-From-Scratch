import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

class GRU:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize a GRU network.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state
            output_size: Dimension of output
        """
        # Xavier/Glorot initialization - scale by sqrt(2 / (fan_in + fan_out))
        # Update gate parameters
        self.W_zx = np.random.randn(hidden_size, input_size) * np.sqrt(2 / (input_size + hidden_size))
        self.W_zh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / (hidden_size + hidden_size))
        self.b_z = np.zeros((hidden_size, 1))
        
        # Reset gate parameters
        self.W_rx = np.random.randn(hidden_size, input_size) * np.sqrt(2 / (input_size + hidden_size))
        self.W_rh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / (hidden_size + hidden_size))
        self.b_r = np.zeros((hidden_size, 1))
        
        # Candidate hidden state parameters
        self.W_x = np.random.randn(hidden_size, input_size) * np.sqrt(2 / (input_size + hidden_size))
        self.W_h = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / (hidden_size + hidden_size))
        self.b = np.zeros((hidden_size, 1))
        
        # Output layer parameters
        self.W_o = np.random.randn(output_size, hidden_size) * np.sqrt(2 / (hidden_size + output_size))
        self.b_o = np.zeros((output_size, 1))
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))  # Clip to avoid overflow
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function."""
        return np.tanh(x)
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Forward pass through the GRU.
        
        Args:
            x: Input of shape (input_size, 1)
            h_prev: Previous hidden state of shape (hidden_size, 1)
            
        Returns:
            tuple: (output, new_hidden_state, cache)
        """
        # Ensure inputs are column vectors
        if x.shape[1] != 1:
            x = x.reshape(-1, 1)
        
        if h_prev.shape[1] != 1:
            h_prev = h_prev.reshape(-1, 1)
            
        # Update gate
        a_z = np.dot(self.W_zx, x) + np.dot(self.W_zh, h_prev) + self.b_z
        z_t = self.sigmoid(a_z)
        
        # Reset gate
        a_r = np.dot(self.W_rx, x) + np.dot(self.W_rh, h_prev) + self.b_r
        r_t = self.sigmoid(a_r)
        
        # Candidate hidden state
        r_h = r_t * h_prev  # Element-wise product
        a_h = np.dot(self.W_x, x) + np.dot(self.W_h, r_h) + self.b
        h_tilde = self.tanh(a_h)
        
        # Final hidden state
        h_t = (1 - z_t) * h_tilde + z_t * h_prev
        
        # Output layer
        a_o = np.dot(self.W_o, h_t) + self.b_o
        y_hat = a_o  # Linear activation for regression
        
        cache = {
            'x': x, 'h_prev': h_prev,
            'a_z': a_z, 'z_t': z_t,
            'a_r': a_r, 'r_t': r_t,
            'r_h': r_h, 'a_h': a_h, 'h_tilde': h_tilde,
            'h_t': h_t, 'a_o': a_o, 'y_hat': y_hat
        }
        
        return y_hat, h_t, cache
    
    def backward(self, y_hat: np.ndarray, y: np.ndarray, cache: Dict, dh_next: np.ndarray = None) -> Dict:
        """
        Backward pass through the GRU.
        
        Args:
            y_hat: Predicted output
            y: Target output
            cache: Cache from forward pass
            dh_next: Gradient from the next time step
            
        Returns:
            dict: Gradients for all parameters
        """
        # Initialize gradients
        dW_zx, dW_zh, db_z = np.zeros_like(self.W_zx), np.zeros_like(self.W_zh), np.zeros_like(self.b_z)
        dW_rx, dW_rh, db_r = np.zeros_like(self.W_rx), np.zeros_like(self.W_rh), np.zeros_like(self.b_r)
        dW_x, dW_h, db = np.zeros_like(self.W_x), np.zeros_like(self.W_h), np.zeros_like(self.b)
        dW_o, db_o = np.zeros_like(self.W_o), np.zeros_like(self.b_o)
        
        x = cache['x']
        h_prev = cache['h_prev']
        z_t = cache['z_t']
        r_t = cache['r_t']
        r_h = cache['r_h']
        h_tilde = cache['h_tilde']
        h_t = cache['h_t']
        
        # Output layer gradients (MSE loss)
        da_o = y_hat - y  # For MSE loss, gradient is simply prediction minus target
        dW_o = np.dot(da_o, h_t.T)
        db_o = da_o
        
        # Initial gradient for hidden state
        dh = np.dot(self.W_o.T, da_o)
        
        # Add gradient from next time step if it exists
        if dh_next is not None:
            dh += dh_next
            
        # Gradient through final hidden state equation
        dh_tilde = dh * (1 - z_t)
        dh_prev_direct = dh * z_t
        dz_t = dh * (h_prev - h_tilde)
        
        # Gradient through tanh
        da_h = dh_tilde * (1 - h_tilde**2)
        
        # Gradients for candidate hidden state parameters
        dW_x = np.dot(da_h, x.T)
        dW_h = np.dot(da_h, r_h.T)
        db = da_h
        
        # Gradient to reset-modulated hidden state
        dr_h = np.dot(self.W_h.T, da_h)
        
        # Gradient to reset gate
        dr_t = dr_h * h_prev
        
        # Gradient through sigmoid for reset gate
        da_r = dr_t * r_t * (1 - r_t)
        dW_rx = np.dot(da_r, x.T)
        dW_rh = np.dot(da_r, h_prev.T)
        db_r = da_r
        
        # Gradient through sigmoid for update gate
        da_z = dz_t * z_t * (1 - z_t)
        dW_zx = np.dot(da_z, x.T)
        dW_zh = np.dot(da_z, h_prev.T)
        db_z = da_z
        
        # Gradient to previous hidden state (composite of all paths)
        dh_prev_via_r = dr_h * r_t
        dh_prev_via_a_r = np.dot(self.W_rh.T, da_r)
        dh_prev_via_a_z = np.dot(self.W_zh.T, da_z)
        
        dh_prev = dh_prev_direct + dh_prev_via_r + dh_prev_via_a_r + dh_prev_via_a_z
        
        # Gradient to input x (not used in this implementation but kept for completeness)
        dx = (np.dot(self.W_zx.T, da_z) + 
              np.dot(self.W_rx.T, da_r) + 
              np.dot(self.W_x.T, da_h))
        
        grads = {
            'dW_zx': dW_zx, 'dW_zh': dW_zh, 'db_z': db_z,
            'dW_rx': dW_rx, 'dW_rh': dW_rh, 'db_r': db_r,
            'dW_x': dW_x, 'dW_h': dW_h, 'db': db,
            'dW_o': dW_o, 'db_o': db_o,
            'dh_prev': dh_prev, 'dx': dx
        }
        
        return grads
    
    def update_parameters(self, grads: Dict, learning_rate: float) -> None:
        """
        Update all parameters using gradient descent.
        
        Args:
            grads: Gradients dictionary
            learning_rate: Learning rate
        """
        self.W_zx -= learning_rate * grads['dW_zx']
        self.W_zh -= learning_rate * grads['dW_zh']
        self.b_z -= learning_rate * grads['db_z']
        
        self.W_rx -= learning_rate * grads['dW_rx']
        self.W_rh -= learning_rate * grads['dW_rh']
        self.b_r -= learning_rate * grads['db_r']
        
        self.W_x -= learning_rate * grads['dW_x']
        self.W_h -= learning_rate * grads['dW_h']
        self.b -= learning_rate * grads['db']
        
        self.W_o -= learning_rate * grads['dW_o']
        self.b_o -= learning_rate * grads['db_o']

    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray, 
              num_epochs: int = 100, 
              learning_rate: float = 0.01, 
              clip_value: float = 5.0,
              sequence_length: int = None) -> Dict:
        """
        Train the GRU on a dataset.
        
        Args:
            X_train: Training input sequences (shape: [n_samples, sequence_length, input_size])
            y_train: Training targets (shape: [n_samples, sequence_length, output_size])
            num_epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            clip_value: Gradient clipping threshold
            sequence_length: Length of sequences to process (can be None to use full sequences)
            
        Returns:
            dict: Training history
        """
        loss_history = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            # Process each sequence in the training data
            for i in range(len(X_train)):
                x_seq = X_train[i]
                y_seq = y_train[i]
                
                # Use only the specified sequence length if provided
                if sequence_length is not None:
                    x_seq = x_seq[:sequence_length]
                    y_seq = y_seq[:sequence_length]
                
                seq_length = len(x_seq)
                h = np.zeros((self.hidden_size, 1))  # Initialize hidden state
                
                # Forward pass through the sequence
                caches = []
                y_preds = []
                
                for t in range(seq_length):
                    x_t = x_seq[t].reshape(-1, 1)
                    y_t = y_seq[t].reshape(-1, 1)
                    
                    y_pred, h, cache = self.forward(x_t, h)
                    y_preds.append(y_pred)
                    caches.append(cache)
                    
                    # Compute loss for this time step
                    loss = 0.5 * np.sum((y_pred - y_t)**2)  # MSE
                    epoch_loss += loss
                
                # Backward pass through the sequence (BPTT)
                dh_next = np.zeros((self.hidden_size, 1))
                
                # Loop backwards through time steps
                for t in reversed(range(seq_length)):
                    y_t = y_seq[t].reshape(-1, 1)
                    y_pred = y_preds[t]
                    cache = caches[t]
                    
                    # Backward pass for this time step
                    grads = self.backward(y_pred, y_t, cache, dh_next)
                    dh_next = grads['dh_prev']
                    
                    # Clip gradients to prevent exploding gradients
                    for key, grad in grads.items():
                        if key.startswith('d'):
                            grads[key] = np.clip(grad, -clip_value, clip_value)
                    
                    # Update parameters
                    self.update_parameters(grads, learning_rate)
            
            # Average loss for this epoch
            avg_loss = epoch_loss / (len(X_train) * seq_length)
            loss_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
                
        return {'loss': loss_history}
    
    def predict(self, X: np.ndarray, sequence_length: int = None) -> np.ndarray:
        """
        Make predictions using the trained GRU.
        
        Args:
            X: Input sequences (shape: [n_samples, sequence_length, input_size])
            sequence_length: Length of sequences to process
            
        Returns:
            np.ndarray: Predictions
        """
        predictions = []
        
        for i in range(len(X)):
            x_seq = X[i]
            
            # Use only the specified sequence length if provided
            if sequence_length is not None:
                x_seq = x_seq[:sequence_length]
                
            seq_length = len(x_seq)
            h = np.zeros((self.hidden_size, 1))  # Initialize hidden state
            seq_preds = []
            
            for t in range(seq_length):
                x_t = x_seq[t].reshape(-1, 1)
                y_pred, h, _ = self.forward(x_t, h)
                seq_preds.append(y_pred.flatten())
                
            predictions.append(np.array(seq_preds))
            
        return np.array(predictions)


def generate_cosine_data(num_sequences: int, 
                         sequence_length: int, 
                         frequency: float = 0.1, 
                         noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cosine wave data for sequence prediction.
    
    Args:
        num_sequences: Number of sequences to generate
        sequence_length: Length of each sequence
        frequency: Frequency of the cosine wave
        noise_level: Standard deviation of Gaussian noise
        
    Returns:
        tuple: (X, y) where:
            X is input sequences with shape [num_sequences, sequence_length, 1]
            y is target sequences with shape [num_sequences, sequence_length, 1]
    """
    X = []
    y = []
    
    for i in range(num_sequences):
        # Generate a random phase shift for variety
        phase_shift = np.random.uniform(0, 2 * np.pi)
        
        # Generate the time points
        time = np.arange(0, sequence_length)
        
        # Generate the pure cosine signal
        signal = np.cos(2 * np.pi * frequency * time + phase_shift)
        
        # Add noise to create the input
        noisy_signal = signal + np.random.normal(0, noise_level, sequence_length)
        
        # For this task, we want to predict the clean signal from the noisy one
        X.append(noisy_signal.reshape(-1, 1))
        y.append(signal.reshape(-1, 1))
    
    return np.array(X), np.array(y)


def plot_results(X: np.ndarray, 
                 y_true: np.ndarray, 
                 y_pred: np.ndarray, 
                 loss_history: List[float], 
                 sequence_idx: int = 0) -> None:
    """
    Plot the original signal, noisy input, predictions, and loss history.
    
    Args:
        X: Input sequences
        y_true: True target sequences
        y_pred: Predicted sequences
        loss_history: Training loss history
        sequence_idx: Index of sequence to plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot the loss history
    plt.subplot(2, 1, 1)
    plt.plot(loss_history)
    plt.title('Training Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot signals and predictions
    plt.subplot(2, 1, 2)
    time = np.arange(len(X[sequence_idx]))
    
    plt.plot(time, X[sequence_idx], 'b-', alpha=0.5, label='Noisy Input')
    plt.plot(time, y_true[sequence_idx], 'g-', label='True Cosine')
    plt.plot(time, y_pred[sequence_idx], 'r--', label='GRU Prediction')
    
    plt.title('Cosine Wave Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('gru_cosine_prediction.png')
    plt.show()


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate cosine wave data
    num_sequences = 10
    sequence_length = 100
    X, y = generate_cosine_data(num_sequences, sequence_length)
    
    # Create and train the GRU model
    input_size = 1
    hidden_size = 20
    output_size = 1
    
    model = GRU(input_size, hidden_size, output_size)
    
    # Training parameters
    num_epochs = 200
    learning_rate = 0.01
    
    # Train the model
    history = model.train(X, y, num_epochs=num_epochs, learning_rate=learning_rate)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Plot the results
    plot_results(X, y, y_pred, history['loss'])


if __name__ == "__main__":
    main()