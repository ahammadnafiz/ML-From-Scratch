# RNN Encoder-Decoder (GRU) implementation from scratch in NumPy
# Based on: "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation" (Cho et al. 2014)

import numpy as np

np.random.seed(42)  # Set seed for reproducibility

# Activation functions and their derivatives
def sigmoid(x):
    # Sigmoid function: σ(x) = 1/(1 + e^(-x))
    # Maps any input to (0,1), useful for gates that need to be in this range
    return 1 / (1 + np.exp(-x))

def softmax(x):
    # Softmax function: softmax(x_i) = e^(x_i) / ∑_j e^(x_j)
    # Converts logits to probabilities that sum to 1
    # Subtracting max(x) for numerical stability to prevent overflow
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def tanh(x):
    # Hyperbolic tangent: tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
    # Maps any input to (-1,1), useful for maintaining the flow of gradients
    return np.tanh(x)

def dtanh(x):
    # Derivative of tanh: d/dx tanh(x) = 1 - tanh^2(x)
    # Used in backpropagation
    return 1 - np.tanh(x)**2

def dsigmoid(x):
    # Derivative of sigmoid: d/dx σ(x) = σ(x)(1 - σ(x))
    # Used in backpropagation
    s = sigmoid(x)
    return s * (1 - s)

class GRUCell:
    def __init__(self, input_size, hidden_size):
        # input_size: dimension of input vector
        # hidden_size: dimension of hidden state vector
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights with small random values to break symmetry
        # Update gate parameters: z_t = σ(W_z·x_t + U_z·h_{t-1})
        self.W_z = np.random.randn(input_size, hidden_size) * 0.1
        self.U_z = np.random.randn(hidden_size, hidden_size) * 0.1

        # Reset gate parameters: r_t = σ(W_r·x_t + U_r·h_{t-1})
        self.W_r = np.random.randn(input_size, hidden_size) * 0.1
        self.U_r = np.random.randn(hidden_size, hidden_size) * 0.1

        # Candidate activation parameters: h̃_t = tanh(W·x_t + U·(r_t⊙h_{t-1}))
        self.W = np.random.randn(input_size, hidden_size) * 0.1
        self.U = np.random.randn(hidden_size, hidden_size) * 0.1

        # Storage for gradients during backpropagation
        self.grads = {}

    def forward(self, x, h_prev):
        # Forward pass of GRU
        # x: input vector of shape (batch_size, input_size)
        # h_prev: previous hidden state of shape (batch_size, hidden_size)
        
        # Update gate: z_t = σ(W_z·x_t + U_z·h_{t-1})
        # Controls how much of the new state is copied from the new candidate state
        z = sigmoid(x @ self.W_z + h_prev @ self.U_z)
        
        # Reset gate: r_t = σ(W_r·x_t + U_r·h_{t-1})
        # Controls how much of the previous state contributes to the new candidate state
        r = sigmoid(x @ self.W_r + h_prev @ self.U_r)
        
        # Candidate activation: h̃_t = tanh(W·x_t + U·(r_t⊙h_{t-1}))
        # New candidate state based on current input and reset-gated previous state
        h_tilde = tanh(x @ self.W + (r * h_prev) @ self.U)
        
        # Final hidden state: h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t
        # Interpolates between previous state and new candidate state
        h = (1 - z) * h_prev + z * h_tilde
        
        # Cache values needed for backpropagation
        cache = (x, h_prev, z, r, h_tilde, h)
        
        return h, cache
    
    def backward(self, dh, cache):
        # Backward pass of GRU
        # dh: gradient with respect to current hidden state
        # cache: stored values from forward pass
        
        x, h_prev, z, r, h_tilde, h = cache
        
        # Gradient with respect to update gate z and candidate state h_tilde
        dh_tilde = dh * z
        dz = dh * (h_tilde - h_prev)
        dh_prev = dh * (1 - z)
        
        # Gradient with respect to candidate state computation
        dh_tilde_tanh = dh_tilde * dtanh(x @ self.W + (r * h_prev) @ self.U)
        
        # Gradients for candidate state parameters
        if 'W' not in self.grads:
            self.grads['W'] = np.zeros_like(self.W)
            self.grads['U'] = np.zeros_like(self.U)
        self.grads['W'] += x.T @ dh_tilde_tanh
        self.grads['U'] += (r * h_prev).T @ dh_tilde_tanh
        
        # Gradient with respect to reset gate r
        dr = dh_tilde_tanh @ self.U.T * h_prev
        
        # Additional gradient to h_prev from candidate computation
        dh_prev += (dh_tilde_tanh @ self.U.T) * r
        
        # Gradient with respect to update gate computation
        dz_sigmoid = dz * dsigmoid(x @ self.W_z + h_prev @ self.U_z)
        
        # Gradients for update gate parameters
        if 'W_z' not in self.grads:
            self.grads['W_z'] = np.zeros_like(self.W_z)
            self.grads['U_z'] = np.zeros_like(self.U_z)
        self.grads['W_z'] += x.T @ dz_sigmoid
        self.grads['U_z'] += h_prev.T @ dz_sigmoid
        
        # Additional gradient to h_prev from update gate
        dh_prev += dz_sigmoid @ self.U_z.T
        
        # Gradient with respect to reset gate computation
        dr_sigmoid = dr * dsigmoid(x @ self.W_r + h_prev @ self.U_r)
        
        # Gradients for reset gate parameters
        if 'W_r' not in self.grads:
            self.grads['W_r'] = np.zeros_like(self.W_r)
            self.grads['U_r'] = np.zeros_like(self.U_r)
        self.grads['W_r'] += x.T @ dr_sigmoid
        self.grads['U_r'] += h_prev.T @ dr_sigmoid
        
        # Additional gradient to h_prev from reset gate
        dh_prev += dr_sigmoid @ self.U_r.T
        
        # Gradient with respect to input x
        dx = dh_tilde_tanh @ self.W.T + dz_sigmoid @ self.W_z.T + dr_sigmoid @ self.W_r.T
        
        return dx, dh_prev

class Encoder:
    def __init__(self, input_size, hidden_size, vocab_size):
        self.hidden_size = hidden_size
        self.embedding = np.random.randn(vocab_size, input_size) * 0.1
        self.gru = GRUCell(input_size, hidden_size)

    def forward(self, input_seq):
        h = np.zeros((1, self.hidden_size))
        cache = []
        for idx in input_seq:
            x = self.embedding[idx].reshape(1, -1)
            h, step_cache = self.gru.forward(x, h)
            cache.append((step_cache, idx))
        return h, cache

    def backward(self, dh, cache):
        dE = np.zeros_like(self.embedding)
        for step_cache, idx in reversed(cache):
            dx, dh = self.gru.backward(dh, step_cache)
            dE[idx] += dx.squeeze(0)
        return dE

class Decoder:
    def __init__(self, input_size, hidden_size, vocab_size):
        self.hidden_size = hidden_size
        self.embedding = np.random.randn(vocab_size, input_size) * 0.1
        self.gru = GRUCell(input_size, hidden_size)
        self.W_o = np.random.randn(hidden_size, vocab_size) * 0.1

    def forward(self, target_seq, context):
        h = context
        outputs = []
        caches = []
        for idx in target_seq:
            x = self.embedding[idx].reshape(1, -1)
            h, step_cache = self.gru.forward(x, h)
            o = h @ self.W_o
            outputs.append(o)
            caches.append((step_cache, x, h))
        return outputs, caches

    def backward(self, outputs, targets, caches):
        dW_o = np.zeros_like(self.W_o)
        dE = np.zeros_like(self.embedding)
        dh = np.zeros((1, self.hidden_size))

        for o, t, (cache, x, h) in reversed(list(zip(outputs, targets, caches))):
            probs = softmax(o)
            do = probs
            do[0, t] -= 1  # dL/dlogits
            dW_o += h.T @ do
            dh += do @ self.W_o.T
            dx, dh = self.gru.backward(dh, cache)
            idx = np.where((self.embedding == x).all(axis=1))[0][0]
            dE[idx] += dx.squeeze(0)

        return dW_o, dE, dh

def cross_entropy_loss(outputs, targets):
    loss = 0
    for o, t in zip(outputs, targets):
        probs = softmax(o)
        loss -= np.log(probs[0, t] + 1e-9)
    return loss / len(outputs)

def update_params(params, grads, lr):
    for name in params:
        params[name] -= lr * grads[name]

# ===== Training Loop =====
vocab_size = 10
embed_size = 8
hidden_size = 16
learning_rate = 0.01

encoder = Encoder(embed_size, hidden_size, vocab_size)
decoder = Decoder(embed_size, hidden_size, vocab_size)

for epoch in range(1000):
    input_seq = [1, 2, 3]
    target_seq = [4, 5, 6]

    # Forward pass
    context, encoder_cache = encoder.forward(input_seq)
    decoder_outputs, decoder_caches = decoder.forward(target_seq, context)
    loss = cross_entropy_loss(decoder_outputs, target_seq)

    # Backward pass
    W_o_grad, dec_embed_grad, dh_context = decoder.backward(decoder_outputs, target_seq, decoder_caches)
    enc_embed_grad = encoder.backward(dh_context, encoder_cache)

    # Parameter update
    update_params({'W_o': decoder.W_o}, {'W_o': W_o_grad}, learning_rate)
    decoder.embedding -= learning_rate * dec_embed_grad
    encoder.embedding -= learning_rate * enc_embed_grad

    for name in decoder.gru.grads:
        update_params({
            name: decoder.gru.__dict__[name]
        }, {
            name: decoder.gru.grads[name]
        }, learning_rate)

    for name in encoder.gru.grads:
        update_params({
            name: encoder.gru.__dict__[name]
        }, {
            name: encoder.gru.grads[name]
        }, learning_rate)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
