# RNN Encoder-Decoder (GRU) implementation from scratch in NumPy
# Based on: "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation" (Cho et al. 2014)

import numpy as np
from typing import Tuple, List, Optional

class GRUEncoderDecoder:
    """
    Complete GRU Encoder-Decoder implementation from scratch using NumPy with Adam optimizer.
    Follows the mathematical formulations provided in the reference document.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, vocab_size: int,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize GRU Encoder-Decoder with random weights and Adam optimizer parameters.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Dimension of hidden states
            output_dim: Dimension of output embeddings  
            vocab_size: Size of vocabulary for output predictions
            beta1: Adam optimizer parameter for first moment decay
            beta2: Adam optimizer parameter for second moment decay
            epsilon: Adam optimizer parameter for numerical stability
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        
        # Adam optimizer parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step for Adam
        
        # Initialize weights using Xavier/Glorot initialization
        self._init_weights()
        
        # Initialize Adam optimizer states
        self._init_adam_states()
        
    def _init_weights(self):
        """Initialize all weights using Xavier/Glorot initialization."""
        
        # Encoder weights
        self.W_r_enc = self._xavier_init(self.hidden_dim, self.input_dim)
        self.W_z_enc = self._xavier_init(self.hidden_dim, self.input_dim)
        self.W_h_enc = self._xavier_init(self.hidden_dim, self.input_dim)
        
        self.U_r_enc = self._xavier_init(self.hidden_dim, self.hidden_dim)
        self.U_z_enc = self._xavier_init(self.hidden_dim, self.hidden_dim)
        self.U_h_enc = self._xavier_init(self.hidden_dim, self.hidden_dim)
        
        self.b_r_enc = np.zeros((self.hidden_dim, 1))
        self.b_z_enc = np.zeros((self.hidden_dim, 1))
        self.b_h_enc = np.zeros((self.hidden_dim, 1))
        
        # Decoder weights
        self.W_r_dec = self._xavier_init(self.hidden_dim, self.output_dim)
        self.W_z_dec = self._xavier_init(self.hidden_dim, self.output_dim)
        self.W_h_dec = self._xavier_init(self.hidden_dim, self.output_dim)
        
        self.U_r_dec = self._xavier_init(self.hidden_dim, self.hidden_dim)
        self.U_z_dec = self._xavier_init(self.hidden_dim, self.hidden_dim)
        self.U_h_dec = self._xavier_init(self.hidden_dim, self.hidden_dim)
        
        self.b_r_dec = np.zeros((self.hidden_dim, 1))
        self.b_z_dec = np.zeros((self.hidden_dim, 1))
        self.b_h_dec = np.zeros((self.hidden_dim, 1))
        
        # Context transformation weights
        self.W_c = self._xavier_init(self.hidden_dim, self.hidden_dim)
        self.b_c = np.zeros((self.hidden_dim, 1))
        
        # Output weights
        self.W_o = self._xavier_init(self.vocab_size, self.hidden_dim)
        self.b_o = np.zeros((self.vocab_size, 1))
        
    def _init_adam_states(self):
        """Initialize Adam optimizer moment estimates for all parameters."""
        
        # First moment estimates (m)
        self.m = {
            # Encoder parameters
            'W_r_enc': np.zeros_like(self.W_r_enc),
            'W_z_enc': np.zeros_like(self.W_z_enc),
            'W_h_enc': np.zeros_like(self.W_h_enc),
            'U_r_enc': np.zeros_like(self.U_r_enc),
            'U_z_enc': np.zeros_like(self.U_z_enc),
            'U_h_enc': np.zeros_like(self.U_h_enc),
            'b_r_enc': np.zeros_like(self.b_r_enc),
            'b_z_enc': np.zeros_like(self.b_z_enc),
            'b_h_enc': np.zeros_like(self.b_h_enc),
            
            # Decoder parameters
            'W_r_dec': np.zeros_like(self.W_r_dec),
            'W_z_dec': np.zeros_like(self.W_z_dec),
            'W_h_dec': np.zeros_like(self.W_h_dec),
            'U_r_dec': np.zeros_like(self.U_r_dec),
            'U_z_dec': np.zeros_like(self.U_z_dec),
            'U_h_dec': np.zeros_like(self.U_h_dec),
            'b_r_dec': np.zeros_like(self.b_r_dec),
            'b_z_dec': np.zeros_like(self.b_z_dec),
            'b_h_dec': np.zeros_like(self.b_h_dec),
            
            # Output and context parameters
            'W_o': np.zeros_like(self.W_o),
            'b_o': np.zeros_like(self.b_o),
            'W_c': np.zeros_like(self.W_c),
            'b_c': np.zeros_like(self.b_c)
        }
        
        # Second moment estimates (v)
        self.v = {
            # Encoder parameters
            'W_r_enc': np.zeros_like(self.W_r_enc),
            'W_z_enc': np.zeros_like(self.W_z_enc),
            'W_h_enc': np.zeros_like(self.W_h_enc),
            'U_r_enc': np.zeros_like(self.U_r_enc),
            'U_z_enc': np.zeros_like(self.U_z_enc),
            'U_h_enc': np.zeros_like(self.U_h_enc),
            'b_r_enc': np.zeros_like(self.b_r_enc),
            'b_z_enc': np.zeros_like(self.b_z_enc),
            'b_h_enc': np.zeros_like(self.b_h_enc),
            
            # Decoder parameters
            'W_r_dec': np.zeros_like(self.W_r_dec),
            'W_z_dec': np.zeros_like(self.W_z_dec),
            'W_h_dec': np.zeros_like(self.W_h_dec),
            'U_r_dec': np.zeros_like(self.U_r_dec),
            'U_z_dec': np.zeros_like(self.U_z_dec),
            'U_h_dec': np.zeros_like(self.U_h_dec),
            'b_r_dec': np.zeros_like(self.b_r_dec),
            'b_z_dec': np.zeros_like(self.b_z_dec),
            'b_h_dec': np.zeros_like(self.b_h_dec),
            
            # Output and context parameters
            'W_o': np.zeros_like(self.W_o),
            'b_o': np.zeros_like(self.b_o),
            'W_c': np.zeros_like(self.W_c),
            'b_c': np.zeros_like(self.b_c)
        }
        
    def _xavier_init(self, n_out: int, n_in: int) -> np.ndarray:
        """Xavier/Glorot weight initialization."""
        bound = np.sqrt(6.0 / (n_in + n_out))
        return np.random.uniform(-bound, bound, (n_out, n_in))
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability."""
        x = np.clip(x, -500, 500)  # Prevent overflow
        return 1.0 / (1.0 + np.exp(-x))
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function."""
        return np.tanh(x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function with numerical stability."""
        x_shifted = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def gru_cell_forward(self, x: np.ndarray, h_prev: np.ndarray, 
                        W_r: np.ndarray, W_z: np.ndarray, W_h: np.ndarray,
                        U_r: np.ndarray, U_z: np.ndarray, U_h: np.ndarray,
                        b_r: np.ndarray, b_z: np.ndarray, b_h: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass through a single GRU cell.
        
        Returns:
            h_t: Hidden state at time t
            cache: Dictionary containing intermediate values for backprop
        """
        # Reset gate: r_t = σ(W_r * x_t + U_r * h_{t-1} + b_r)
        r_pre = W_r @ x + U_r @ h_prev + b_r
        r_t = self.sigmoid(r_pre)
        
        # Update gate: z_t = σ(W_z * x_t + U_z * h_{t-1} + b_z)
        z_pre = W_z @ x + U_z @ h_prev + b_z
        z_t = self.sigmoid(z_pre)
        
        # Candidate hidden state: h̃_t = tanh(W_h * x_t + U_h * (r_t ⊙ h_{t-1}) + b_h)
        h_tilde_pre = W_h @ x + U_h @ (r_t * h_prev) + b_h
        h_tilde = self.tanh(h_tilde_pre)
        
        # Final hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        # Cache for backpropagation
        cache = {
            'x': x, 'h_prev': h_prev, 'h_t': h_t,
            'r_pre': r_pre, 'r_t': r_t,
            'z_pre': z_pre, 'z_t': z_t,
            'h_tilde_pre': h_tilde_pre, 'h_tilde': h_tilde
        }
        
        return h_t, cache
    
    def encoder_forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Encoder forward pass.
        
        Args:
            X: Input sequence of shape (input_dim, T)
            
        Returns:
            context: Context vector (final encoder hidden state)
            caches: List of cache dictionaries for each time step
        """
        T = X.shape[1]
        h = np.zeros((self.hidden_dim, 1))
        caches = []
        
        for t in range(T):
            x_t = X[:, t:t+1]
            h, cache = self.gru_cell_forward(
                x_t, h,
                self.W_r_enc, self.W_z_enc, self.W_h_enc,
                self.U_r_enc, self.U_z_enc, self.U_h_enc,
                self.b_r_enc, self.b_z_enc, self.b_h_enc
            )
            caches.append(cache)
        
        context = h
        return context, caches
    
    def decoder_forward(self, Y: np.ndarray, context: np.ndarray) -> Tuple[np.ndarray, List[dict], dict]:
        """
        Decoder forward pass.
        
        Args:
            Y: Target sequence of shape (output_dim, S)
            context: Context vector from encoder
            
        Returns:
            predictions: Output predictions of shape (vocab_size, S)
            caches: List of cache dictionaries for each time step
            init_cache: Cache for context transformation
        """
        S = Y.shape[1]
        
        # Initialize decoder hidden state: h_0 = tanh(W_c * c + b_c)
        h_0_pre = self.W_c @ context + self.b_c
        h = self.tanh(h_0_pre)
        
        init_cache = {
            'context': context,
            'h_0_pre': h_0_pre,
            'h_0': h
        }
        
        caches = []
        predictions = np.zeros((self.vocab_size, S))
        
        for t in range(S):
            # Use previous target as input (teacher forcing)
            if t == 0:
                y_prev = np.zeros((self.output_dim, 1))  # Start token
            else:
                y_prev = Y[:, t-1:t]
            
            h, cache = self.gru_cell_forward(
                y_prev, h,
                self.W_r_dec, self.W_z_dec, self.W_h_dec,
                self.U_r_dec, self.U_z_dec, self.U_h_dec,
                self.b_r_dec, self.b_z_dec, self.b_h_dec
            )
            
            # Output: o_t = W_o * h_t + b_o, p_t = softmax(o_t)
            o_t = self.W_o @ h + self.b_o
            p_t = self.softmax(o_t)
            
            predictions[:, t:t+1] = p_t
            cache['o_t'] = o_t
            cache['p_t'] = p_t
            caches.append(cache)
        
        return predictions, caches, init_cache
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Predicted probabilities of shape (vocab_size, S)
            targets: One-hot target vectors of shape (vocab_size, S)
            
        Returns:
            loss: Cross-entropy loss value
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        loss = -np.sum(targets * np.log(predictions))
        return loss
    
    def gru_cell_backward(self, dh_t: np.ndarray, cache: dict,
                         W_r: np.ndarray, W_z: np.ndarray, W_h: np.ndarray,
                         U_r: np.ndarray, U_z: np.ndarray, U_h: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Backward pass through a single GRU cell.
        
        Returns:
            dh_prev: Gradient w.r.t. previous hidden state
            grads: Dictionary containing parameter gradients
        """
        x = cache['x']
        h_prev = cache['h_prev']
        r_t = cache['r_t']
        z_t = cache['z_t']
        h_tilde = cache['h_tilde']
        
        # Gradient w.r.t. candidate hidden state
        dh_tilde = dh_t * z_t
        
        # Gradient w.r.t. update gate
        dz_t = dh_t * (h_tilde - h_prev)
        
        # Gradient w.r.t. reset gate (through candidate hidden state)
        dh_tilde_pre = dh_tilde * (1 - h_tilde**2)  # tanh derivative
        dr_t = (dh_tilde_pre.T @ U_h).T * h_prev
        
        # Pre-activation gradients
        dz_pre = dz_t * z_t * (1 - z_t)  # sigmoid derivative
        dr_pre = dr_t * r_t * (1 - r_t)  # sigmoid derivative
        
        # Parameter gradients
        dW_z = dz_pre @ x.T
        dU_z = dz_pre @ h_prev.T
        db_z = dz_pre
        
        dW_r = dr_pre @ x.T
        dU_r = dr_pre @ h_prev.T
        db_r = dr_pre
        
        dW_h = dh_tilde_pre @ x.T
        dU_h = dh_tilde_pre @ (r_t * h_prev).T
        db_h = dh_tilde_pre
        
        # Gradient w.r.t. previous hidden state
        dh_prev = (dz_pre.T @ U_z).T + (dr_pre.T @ U_r).T + \
                  (dh_tilde_pre.T @ U_h).T * r_t + dh_t * (1 - z_t)
        
        # Gradient w.r.t. input x
        dx = (dz_pre.T @ W_z).T + (dr_pre.T @ W_r).T + (dh_tilde_pre.T @ W_h).T
        
        grads = {
            'dW_z': dW_z, 'dU_z': dU_z, 'db_z': db_z,
            'dW_r': dW_r, 'dU_r': dU_r, 'db_r': db_r,
            'dW_h': dW_h, 'dU_h': dU_h, 'db_h': db_h,
            'dx': dx
        }
        
        return dh_prev, grads
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray,
                 enc_caches: List[dict], dec_caches: List[dict], 
                 init_cache: dict) -> dict:
        """
        Complete backward pass through encoder-decoder.
        
        Returns:
            all_grads: Dictionary containing all parameter gradients
        """
        S = len(dec_caches)
        T = len(enc_caches)
        
        # Initialize gradient accumulators
        all_grads = {
            # Encoder gradients
            'dW_r_enc': np.zeros_like(self.W_r_enc),
            'dW_z_enc': np.zeros_like(self.W_z_enc),
            'dW_h_enc': np.zeros_like(self.W_h_enc),
            'dU_r_enc': np.zeros_like(self.U_r_enc),
            'dU_z_enc': np.zeros_like(self.U_z_enc),
            'dU_h_enc': np.zeros_like(self.U_h_enc),
            'db_r_enc': np.zeros_like(self.b_r_enc),
            'db_z_enc': np.zeros_like(self.b_z_enc),
            'db_h_enc': np.zeros_like(self.b_h_enc),
            
            # Decoder gradients
            'dW_r_dec': np.zeros_like(self.W_r_dec),
            'dW_z_dec': np.zeros_like(self.W_z_dec),
            'dW_h_dec': np.zeros_like(self.W_h_dec),
            'dU_r_dec': np.zeros_like(self.U_r_dec),
            'dU_z_dec': np.zeros_like(self.U_z_dec),
            'dU_h_dec': np.zeros_like(self.U_h_dec),
            'db_r_dec': np.zeros_like(self.b_r_dec),
            'db_z_dec': np.zeros_like(self.b_z_dec),
            'db_h_dec': np.zeros_like(self.b_h_dec),
            
            # Output and context gradients
            'dW_o': np.zeros_like(self.W_o),
            'db_o': np.zeros_like(self.b_o),
            'dW_c': np.zeros_like(self.W_c),
            'db_c': np.zeros_like(self.b_c)
        }
        
        # Output layer gradients: ∂L/∂o_t = p_t - y_t
        do = predictions - targets
        
        # Accumulate output layer gradients
        for t in range(S):
            h_t_dec = dec_caches[t]['h_t']
            all_grads['dW_o'] += do[:, t:t+1] @ h_t_dec.T
            all_grads['db_o'] += do[:, t:t+1]
        
        # Decoder backward pass
        dh_dec = np.zeros((self.hidden_dim, 1))
        
        for t in reversed(range(S)):
            # Gradient from output layer
            dh_t_output = self.W_o.T @ do[:, t:t+1]
            dh_t_total = dh_t_output + dh_dec
            
            # Backward through GRU cell
            dh_prev, grads = self.gru_cell_backward(
                dh_t_total, dec_caches[t],
                self.W_r_dec, self.W_z_dec, self.W_h_dec,
                self.U_r_dec, self.U_z_dec, self.U_h_dec
            )
            
            # Accumulate gradients (skip dx which is not a parameter)
            for key, grad in grads.items():
                if key == 'dx':
                    continue
                all_grads[key.replace('dW_z', 'dW_z_dec').replace('dW_r', 'dW_r_dec').replace('dW_h', 'dW_h_dec')
                          .replace('dU_z', 'dU_z_dec').replace('dU_r', 'dU_r_dec').replace('dU_h', 'dU_h_dec')
                          .replace('db_z', 'db_z_dec').replace('db_r', 'db_r_dec').replace('db_h', 'db_h_dec')] += grad
            
            dh_dec = dh_prev
        
        # Context transformation gradients
        dh_0 = dh_dec
        dh_0_pre = dh_0 * (1 - init_cache['h_0']**2)  # tanh derivative
        
        all_grads['dW_c'] = dh_0_pre @ init_cache['context'].T
        all_grads['db_c'] = dh_0_pre
        
        # Gradient w.r.t. context (encoder final state)
        dc = self.W_c.T @ dh_0_pre
        
        # Encoder backward pass
        dh_enc = dc
        
        for t in reversed(range(T)):
            dh_prev, grads = self.gru_cell_backward(
                dh_enc, enc_caches[t],
                self.W_r_enc, self.W_z_enc, self.W_h_enc,
                self.U_r_enc, self.U_z_enc, self.U_h_enc
            )
            
            # Accumulate gradients (skip dx which is not a parameter)
            for key, grad in grads.items():
                if key == 'dx':
                    continue
                all_grads[key.replace('dW_z', 'dW_z_enc').replace('dW_r', 'dW_r_enc').replace('dW_h', 'dW_h_enc')
                          .replace('dU_z', 'dU_z_enc').replace('dU_r', 'dU_r_enc').replace('dU_h', 'dU_h_enc')
                          .replace('db_z', 'db_z_enc').replace('db_r', 'db_r_enc').replace('db_h', 'db_h_enc')] += grad
            
            dh_enc = dh_prev
        
        return all_grads
    
    def clip_gradients(self, gradients: dict, max_norm: float = 1.0) -> dict:
        """Clip gradients to prevent exploding gradients."""
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += np.sum(grad**2)
        total_norm = np.sqrt(total_norm)
        
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-8)
            for key in gradients:
                gradients[key] *= clip_coef
        
        return gradients
    
    def adam_update(self, gradients: dict, learning_rate: float = 0.001):
        """
        Update parameters using Adam optimizer.
        
        Adam update rule:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
        """
        self.t += 1  # Increment time step
        
        # Bias correction factors
        bias_correction1 = 1 - self.beta1**self.t
        bias_correction2 = 1 - self.beta2**self.t
        
        # Parameter mapping for easier updates
        param_map = {
            'dW_r_enc': 'W_r_enc', 'dW_z_enc': 'W_z_enc', 'dW_h_enc': 'W_h_enc',
            'dU_r_enc': 'U_r_enc', 'dU_z_enc': 'U_z_enc', 'dU_h_enc': 'U_h_enc',
            'db_r_enc': 'b_r_enc', 'db_z_enc': 'b_z_enc', 'db_h_enc': 'b_h_enc',
            'dW_r_dec': 'W_r_dec', 'dW_z_dec': 'W_z_dec', 'dW_h_dec': 'W_h_dec',
            'dU_r_dec': 'U_r_dec', 'dU_z_dec': 'U_z_dec', 'dU_h_dec': 'U_h_dec',
            'db_r_dec': 'b_r_dec', 'db_z_dec': 'b_z_dec', 'db_h_dec': 'b_h_dec',
            'dW_o': 'W_o', 'db_o': 'b_o', 'dW_c': 'W_c', 'db_c': 'b_c'
        }
        
        # Update each parameter using Adam
        for grad_key, param_key in param_map.items():
            if grad_key in gradients:
                g = gradients[grad_key]  # Current gradient
                
                # Update biased first and second moment estimates
                self.m[param_key] = self.beta1 * self.m[param_key] + (1 - self.beta1) * g
                self.v[param_key] = self.beta2 * self.v[param_key] + (1 - self.beta2) * (g**2)
                
                # Compute bias-corrected first and second moment estimates
                m_hat = self.m[param_key] / bias_correction1
                v_hat = self.v[param_key] / bias_correction2
                
                # Update parameter
                param = getattr(self, param_key)
                param -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def update_parameters(self, gradients: dict, learning_rate: float = 0.001):
        """Update parameters using Adam optimizer (wrapper for backward compatibility)."""
        self.adam_update(gradients, learning_rate)
    
    def train_step(self, X: np.ndarray, Y: np.ndarray, targets: np.ndarray, 
                   learning_rate: float = 0.001) -> float:
        """
        Complete training step: forward pass, loss computation, backward pass, parameter update.
        
        Args:
            X: Input sequence (input_dim, T)
            Y: Target sequence for decoder input (output_dim, S)
            targets: One-hot target vectors (vocab_size, S)
            learning_rate: Learning rate for parameter updates
            
        Returns:
            loss: Cross-entropy loss value
        """
        # Forward pass
        context, enc_caches = self.encoder_forward(X)
        predictions, dec_caches, init_cache = self.decoder_forward(Y, context)
        
        # Compute loss
        loss = self.compute_loss(predictions, targets)
        
        # Backward pass
        gradients = self.backward(predictions, targets, enc_caches, dec_caches, init_cache)
        
        # Check gradients
        if not self.check_gradients(gradients):
            print("Gradient check failed - skipping update")
            return loss
        
        # Clip gradients
        gradients = self.clip_gradients(gradients)
        
        # Update parameters using Adam
        self.adam_update(gradients, learning_rate)
        
        return loss
    
    def predict(self, X: np.ndarray, max_length: int = 20) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Input sequence (input_dim, T)
            max_length: Maximum length of generated sequence
            
        Returns:
            Generated sequence of token indices
        """
        # Encode input
        context, _ = self.encoder_forward(X)
        
        # Initialize decoder
        h_0_pre = self.W_c @ context + self.b_c
        h = self.tanh(h_0_pre)
        
        generated = []
        y_prev = np.zeros((self.output_dim, 1))  # Start token
        
        for _ in range(max_length):
            # Decoder step
            h, _ = self.gru_cell_forward(
                y_prev, h,
                self.W_r_dec, self.W_z_dec, self.W_h_dec,
                self.U_r_dec, self.U_z_dec, self.U_h_dec,
                self.b_r_dec, self.b_z_dec, self.b_h_dec
            )
            
            # Generate output
            o_t = self.W_o @ h + self.b_o
            p_t = self.softmax(o_t)
            
            # Sample next token (greedy decoding)
            next_token = np.argmax(p_t)
            generated.append(next_token)
            
            # Update input for next step (simplified - would need proper embedding)
            y_prev = np.zeros((self.output_dim, 1))
            if next_token < self.output_dim:
                y_prev[next_token, 0] = 1.0
            
            # Note: Removed assumption that token 0 is end token since our data
            # generation pattern includes token 0 as a valid target token
        
        return np.array(generated)

    def check_gradients(self, gradients: dict) -> bool:
        """Check if gradients are reasonable (not NaN or too small/large)."""
        for key, grad in gradients.items():
            if np.isnan(grad).any() or np.isinf(grad).any():
                print(f"Warning: NaN or Inf found in gradient {key}")
                return False
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                print(f"Warning: Very small gradient norm for {key}: {grad_norm}")
            elif grad_norm > 100:
                print(f"Warning: Very large gradient norm for {key}: {grad_norm}")
        return True

    def lr_schedule(self, epoch: int, initial_lr: float = 0.001) -> float:
        """
        Learning rate schedule with decay.
        
        Args:
            epoch: Current epoch number
            initial_lr: Initial learning rate
            
        Returns:
            Scheduled learning rate
        """
        # Exponential decay: lr = initial_lr * 0.95^(epoch/10)
        decay_rate = 0.95
        decay_steps = 10
        return initial_lr * (decay_rate ** (epoch // decay_steps))
    
    def generate_structured_data(self, seq_len: int = 10, target_len: int = 5, batch_size: int = 1):
        """
        Generate more structured training data for better learning.
        
        Args:
            seq_len: Length of input sequence
            target_len: Length of target sequence  
            batch_size: Number of sequences in batch
            
        Returns:
            Tuple of (input_sequences, decoder_inputs, targets)
        """
        # Create input sequences with some structure (e.g., repeating patterns)
        X = np.zeros((self.input_dim, seq_len))  # Correct dimensions: (input_dim, seq_len)
        Y = np.zeros((self.output_dim, target_len))  # Correct dimensions: (output_dim, target_len)
        targets = np.zeros((self.vocab_size, target_len))  # Correct dimensions: (vocab_size, target_len)
        
        # Input: alternating pattern
        for t in range(seq_len):
            idx = (t % min(4, self.input_dim-1)) + 1  # Pattern: 1,2,3,4,1,2,3,4... (ensure within bounds)
            if idx < self.input_dim:
                X[idx, t] = 1.0
            
        # Target: related to input pattern but shorter
        for t in range(target_len):
            idx = ((t * 2) % self.vocab_size)  # Different pattern for output
            if idx % self.output_dim < self.output_dim:
                Y[idx % self.output_dim, t] = 1.0
            targets[idx, t] = 1.0
            
        return X, Y, targets

    def evaluate_model(self, X: np.ndarray, Y: np.ndarray, targets: np.ndarray) -> dict:
        """
        Evaluate model performance with detailed metrics.
        
        Args:
            X: Input sequence
            Y: Decoder input sequence
            targets: Target outputs
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Forward pass
        context, enc_caches = self.encoder_forward(X)
        predictions, dec_caches, init_cache = self.decoder_forward(Y, context)
        
        # Compute loss
        loss = self.compute_loss(predictions, targets)
        
        # Compute accuracy (proportion of correctly predicted tokens)
        pred_classes = np.argmax(predictions, axis=0)
        target_classes = np.argmax(targets, axis=0)
        accuracy = np.mean(pred_classes == target_classes)
        
        # Compute perplexity
        perplexity = np.exp(loss)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
            'predictions': predictions,
            'pred_classes': pred_classes,
            'target_classes': target_classes
        }


# Example usage and testing with ALL methods
if __name__ == "__main__":
    print("=" * 80)
    print("GRU ENCODER-DECODER COMPREHENSIVE EXAMPLE")
    print("Demonstrating ALL methods including previously unused ones")
    print("=" * 80)
    
    # Initialize model with Adam optimizer
    model = GRUEncoderDecoder(
        input_dim=10,
        hidden_dim=20,
        output_dim=10,
        vocab_size=50,
        beta1=0.9,      # Adam parameter
        beta2=0.999,    # Adam parameter
        epsilon=1e-8    # Adam parameter
    )
    
    print("✓ Model initialized successfully!")
    print(f"Adam parameters: β₁={model.beta1}, β₂={model.beta2}, ε={model.epsilon}")
    
    # 1. USE GENERATE_STRUCTURED_DATA METHOD (Previously unused)
    print(f"\n1. USING generate_structured_data() method:")
    print("-" * 50)
    
    X, Y, targets = model.generate_structured_data(seq_len=12, target_len=8, batch_size=1)
    print(f"✓ Structured data generated!")
    print(f"  Input shape: {X.shape}")
    print(f"  Decoder input shape: {Y.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Input pattern: {np.argmax(X, axis=0)}")
    print(f"  Target pattern: {np.argmax(targets, axis=0)}")
    
    # 2. USE EVALUATE_MODEL METHOD (Previously unused)
    print(f"\n2. USING evaluate_model() method:")
    print("-" * 50)
    
    initial_eval = model.evaluate_model(X, Y, targets)
    print(f"✓ Initial evaluation completed!")
    print(f"  Loss: {initial_eval['loss']:.4f}")
    print(f"  Accuracy: {initial_eval['accuracy']:.4f}")
    print(f"  Perplexity: {initial_eval['perplexity']:.4f}")
    
    # 3. USE LR_SCHEDULE METHOD (Previously unused) 
    print(f"\n3. USING lr_schedule() method:")
    print("-" * 50)
    
    print("Learning rate schedule over epochs:")
    for epoch in [0, 5, 10, 20, 50, 100]:
        lr = model.lr_schedule(epoch, initial_lr=0.01)
        print(f"  Epoch {epoch:3d}: lr = {lr:.6f}")
    
    # 4. Training with adaptive learning rate and evaluation
    print(f"\n4. TRAINING WITH ADAPTIVE LEARNING RATE:")
    print("-" * 50)
    
    print("Training with lr_schedule, generate_structured_data, and evaluate_model...")
    
    # Generate validation data using generate_structured_data
    X_val, Y_val, targets_val = model.generate_structured_data(seq_len=12, target_len=8)
    
    losses = []
    accuracies = []
    val_accuracies = []
    learning_rates = []
    
    num_epochs = 1000
    initial_lr = 0.01
    
    for epoch in range(num_epochs):
        # Use lr_schedule method for adaptive learning rate
        current_lr = model.lr_schedule(epoch, initial_lr)
        learning_rates.append(current_lr)
        
        # Training step
        loss = model.train_step(X, Y, targets, learning_rate=current_lr)
        losses.append(loss)
        
        # Evaluate using evaluate_model method
        if epoch % 10 == 0:
            train_eval = model.evaluate_model(X, Y, targets)
            val_eval = model.evaluate_model(X_val, Y_val, targets_val)
            
            accuracies.append(train_eval['accuracy'])
            val_accuracies.append(val_eval['accuracy'])
            
            print(f"Epoch {epoch+1:3d}: Loss={loss:.4f}, "
                  f"Train Acc={train_eval['accuracy']:.4f}, "
                  f"Val Acc={val_eval['accuracy']:.4f}, "
                  f"LR={current_lr:.6f}")
    
    # 5. Final comprehensive evaluation
    print(f"\n5. FINAL EVALUATION:")
    print("-" * 50)
    
    final_eval = model.evaluate_model(X, Y, targets)
    print(f"✓ Final evaluation using evaluate_model():")
    print(f"  Final loss: {final_eval['loss']:.4f}")
    print(f"  Final accuracy: {final_eval['accuracy']:.4f}")
    print(f"  Final perplexity: {final_eval['perplexity']:.4f}")
    print(f"  Improvement: {((initial_eval['loss'] - final_eval['loss']) / initial_eval['loss'] * 100):+.2f}% loss reduction")
    
    # 6. Test prediction capability
    print(f"\n6. PREDICTION TESTING:")
    print("-" * 50)
    
    # Generate test data using generate_structured_data
    X_test, _, _ = model.generate_structured_data(seq_len=10, target_len=6)
    generated = model.predict(X_test, max_length=8)
    print(f"✓ Prediction completed!")
    print(f"  Test input pattern: {np.argmax(X_test, axis=0)}")
    print(f"  Generated sequence: {generated}")
    
    # 7. Demonstrate gradient checking and clipping
    print(f"\n7. GRADIENT ANALYSIS:")
    print("-" * 50)
    
    # Forward pass to get gradients
    context, enc_caches = model.encoder_forward(X)
    predictions, dec_caches, init_cache = model.decoder_forward(Y, context)
    gradients = model.backward(predictions, targets, enc_caches, dec_caches, init_cache)
    
    # Use check_gradients method
    gradient_check = model.check_gradients(gradients)
    print(f"✓ Gradient check: {'PASSED' if gradient_check else 'FAILED'}")
    
    # Use clip_gradients method
    clipped_grads = model.clip_gradients(gradients.copy(), max_norm=1.0)
    print(f"✓ Gradient clipping applied")
    
    # 8. Generate multiple datasets for comparison
    print(f"\n8. MULTIPLE DATASET GENERATION:")
    print("-" * 50)
    
    print("Generating different structured datasets:")
    for i, (seq_len, target_len) in enumerate([(8, 5), (12, 8), (16, 10)]):
        X_i, Y_i, targets_i = model.generate_structured_data(seq_len, target_len)
        eval_i = model.evaluate_model(X_i, Y_i, targets_i)
        print(f"  Dataset {i+1}: SeqLen={seq_len}, TargetLen={target_len}, "
              f"Loss={eval_i['loss']:.4f}, Acc={eval_i['accuracy']:.4f}")
    
    # 9. Summary of all methods used
    print(f"\n9. METHODS UTILIZATION SUMMARY:")
    print("-" * 50)
    
    print("✓ ALL methods have been demonstrated:")
    print("  Core methods:")
    print("    • encoder_forward() - Used in training and evaluation")
    print("    • decoder_forward() - Used in training and evaluation") 
    print("    • compute_loss() - Used in training steps")
    print("    • backward() - Used in training steps")
    print("    • adam_update() - Used in training steps")
    print("    • train_step() - Used in training loop")
    print("    • predict() - Used for sequence generation")
    print("    • check_gradients() - Used for gradient validation")
    print("    • clip_gradients() - Used for gradient stability")
    print("")
    print("  Previously unused methods:")
    print("    • lr_schedule() - Used for adaptive learning rates")
    print("    • generate_structured_data() - Used for data generation")
    print("    • evaluate_model() - Used for comprehensive evaluation")
    
    print(f"\nFinal Statistics:")
    print(f"  - Total epochs trained: {num_epochs}")
    print(f"  - Final learning rate: {learning_rates[-1]:.6f}")
    print(f"  - Adam optimizer steps: {model.t}")
    print(f"  - Loss improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
    
    print("\n" + "=" * 80)
    print("GRU ENCODER-DECODER COMPREHENSIVE DEMONSTRATION COMPLETED!")
    print("All methods including previously unused ones have been utilized.")
    print("=" * 80)