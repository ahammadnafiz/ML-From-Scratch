import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Data Generation
# ----------------------
def generate_sine_wave(time_steps, sequence_length):
    t = np.linspace(0, 4*np.pi, time_steps)
    data = np.sin(t)
    
    # Create sequences
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    
    return np.array(X), np.array(y)

# ----------------------
# LSTM Implementation
# ----------------------
class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        # Gates parameters
        self.W_f = np.random.randn(hidden_size, input_size) * 0.01
        self.U_f = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1))
        
        self.W_i = np.random.randn(hidden_size, input_size) * 0.01
        self.U_i = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))
        
        self.W_o = np.random.randn(hidden_size, input_size) * 0.01
        self.U_o = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))
        
        self.W_c = np.random.randn(hidden_size, input_size) * 0.01
        self.U_c = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_c = np.zeros((hidden_size, 1))
        
        # Output layer
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))
        
    def forward(self, x_seq):
        sequence_length = len(x_seq)
        self.cache = []
        h = np.zeros((self.W_f.shape[0], 1))
        c = np.zeros((self.W_f.shape[0], 1))
        
        for t in range(sequence_length):
            x = x_seq[t].reshape(-1, 1)
            
            # Gates
            a_f = self.W_f @ x + self.U_f @ h + self.b_f
            f = sigmoid(a_f)
            
            a_i = self.W_i @ x + self.U_i @ h + self.b_i
            i = sigmoid(a_i)
            
            a_o = self.W_o @ x + self.U_o @ h + self.b_o
            o = sigmoid(a_o)
            
            # Cell candidate
            a_c = self.W_c @ x + self.U_c @ h + self.b_c
            tilde_c = np.tanh(a_c)
            
            # Cell state
            c = f * c + i * tilde_c
            
            # Hidden state
            h = o * np.tanh(c)
            
            # Output
            y_hat = self.W_y @ h + self.b_y
            
            self.cache.append({
                'x': x, 'h_prev': h.copy(), 'c_prev': c.copy(),
                'f': f, 'i': i, 'o': o, 'tilde_c': tilde_c,
                'c': c.copy(), 'h': h.copy(), 'y_hat': y_hat
            })
            
        return h, y_hat
    
    def backward(self, dy):
        params = self.__dict__
        gradients = {key: np.zeros_like(value) for key, value in params.items() 
                    if key not in ['cache']}
        
        dh_next = np.zeros_like(self.cache[0]['h'])
        dc_next = np.zeros_like(self.cache[0]['c'])
        
        for t in reversed(range(len(self.cache))):
            current = self.cache[t]
            
            # Output layer gradients
            # Use dy directly if we're at the last timestep, otherwise no direct gradient from output
            if t == len(self.cache) - 1:
                dy_hat = dy
            else:
                dy_hat = np.zeros_like(self.cache[-1]['y_hat'])
                
            dW_y = dy_hat @ current['h'].T
            db_y = dy_hat
            dh = self.W_y.T @ dy_hat + dh_next
            
            # Output gate
            do = dh * np.tanh(current['c'])
            da_o = do * current['o'] * (1 - current['o'])
            
            # Cell state
            dc = dh * current['o'] * (1 - np.tanh(current['c'])**2)
            dc += dc_next
            
            # Gates and cell candidate
            df = dc * current['c_prev']
            di = dc * current['tilde_c']
            dtilde_c = dc * current['i']
            dc_prev = dc * current['f']
            
            # Input gate
            da_i = di * current['i'] * (1 - current['i'])
            
            # Forget gate
            da_f = df * current['f'] * (1 - current['f'])
            
            # Cell candidate
            da_c = dtilde_c * (1 - current['tilde_c']**2)
            
            # Parameter gradients
            gradients['W_f'] += da_f @ current['x'].T
            gradients['U_f'] += da_f @ current['h_prev'].T
            gradients['b_f'] += da_f
            
            gradients['W_i'] += da_i @ current['x'].T
            gradients['U_i'] += da_i @ current['h_prev'].T
            gradients['b_i'] += da_i
            
            gradients['W_o'] += da_o @ current['x'].T
            gradients['U_o'] += da_o @ current['h_prev'].T
            gradients['b_o'] += da_o
            
            gradients['W_c'] += da_c @ current['x'].T
            gradients['U_c'] += da_c @ current['h_prev'].T
            gradients['b_c'] += da_c
            
            gradients['W_y'] += dW_y
            gradients['b_y'] += db_y
            
            # Previous hidden state
            dh_prev = (
                self.U_f.T @ da_f +
                self.U_i.T @ da_i +
                self.U_o.T @ da_o +
                self.U_c.T @ da_c
            )
            
            dh_next = dh_prev
            dc_next = dc_prev
            
        return gradients
    
    def update(self, gradients, lr):
        for key in gradients:
            if key.startswith('W') or key.startswith('U') or key.startswith('b'):
                self.__dict__[key] -= lr * gradients[key]

# ----------------------
# Utility Functions
# ----------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# ----------------------
# Training
# ----------------------
# Hyperparameters
SEQ_LENGTH = 20
HIDDEN_SIZE = 64
EPOCHS = 50
LR = 0.01

# Generate data
X, y = generate_sine_wave(1000, SEQ_LENGTH)
X = X.reshape(-1, SEQ_LENGTH, 1)
y = y.reshape(-1, 1)

# Initialize LSTM
lstm = LSTM(input_size=1, hidden_size=HIDDEN_SIZE, output_size=1)

# Training loop
losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i in range(len(X)):
        # Forward pass
        _, y_hat = lstm.forward(X[i])
        
        # Compute loss
        loss = mse_loss(y[i], y_hat)
        epoch_loss += loss
        
        # Backward pass
        dy = (y_hat - y[i]).reshape(1, 1)
        gradients = lstm.backward(dy)
        
        # Update parameters
        lstm.update(gradients, LR)
    
    avg_loss = epoch_loss / len(X)
    losses.append(avg_loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

# ----------------------
# Prediction & Visualization
# ----------------------
# Generate test data
test_X, test_y = generate_sine_wave(200, SEQ_LENGTH)
test_X = test_X.reshape(-1, SEQ_LENGTH, 1)

# Make predictions
predictions = []
for seq in test_X:
    _, y_hat = lstm.forward(seq)
    predictions.append(y_hat[0,0])

# Calculate prediction metrics
mse = np.mean((np.array(predictions) - test_y)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(np.array(predictions) - test_y))

# Plot results
plt.figure(figsize=(14, 8))
plt.plot(test_y, label='True values', linewidth=2)
plt.plot(predictions, label=f'Predictions (MSE: {mse:.4f})', linestyle='--', linewidth=2)
plt.title("Sine Wave Prediction using LSTM", fontsize=16)
plt.xlabel("Time steps", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add metrics annotation
plt.annotate(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}",
            xy=(0.02, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
            fontsize=12)

# Save the prediction plot
plt.savefig('/media/nafiz/NewVolume/ML-From-Scratch/Sequence Modeling/LSTM/lstm_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(losses, linewidth=2, color='#1f77b4')
plt.title("LSTM Training Loss", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Mean Squared Error", fontsize=12)
plt.grid(True, alpha=0.3)

# Add more visual elements to training loss plot
plt.axhline(y=min(losses), color='r', linestyle='--', alpha=0.7, 
           label=f'Min Loss: {min(losses):.6f}')
plt.annotate(f'Min Loss: {min(losses):.6f} at epoch {np.argmin(losses)}',
            xy=(np.argmin(losses), min(losses)),
            xytext=(np.argmin(losses)+5, min(losses)*1.5),
            arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.5),
            fontsize=10)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the loss plot
plt.savefig('/media/nafiz/NewVolume/ML-From-Scratch/Sequence Modeling/LSTM/lstm_training_loss.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a third combined visualization showing both training and validation
plt.figure(figsize=(16, 12))

# Subplot 1: Predictions vs Ground Truth
plt.subplot(2, 1, 1)
plt.plot(test_y, label='True values', linewidth=2)
plt.plot(predictions, label=f'Predictions', linestyle='--', linewidth=2)
plt.title("LSTM Sine Wave Prediction Performance", fontsize=16)
plt.xlabel("Time steps", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Add metrics box
plt.text(0.02, 0.85, 
         f"Model Performance:\n"\
         f"MSE: {mse:.6f}\n"\
         f"RMSE: {rmse:.6f}\n"\
         f"MAE: {mae:.6f}",
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="black", alpha=0.8),
         fontsize=12)

# Subplot 2: Training Loss
plt.subplot(2, 1, 2)
plt.plot(losses, linewidth=2, color='#1f77b4')
plt.title("Training Loss Progression", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Mean Squared Error", fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=min(losses), color='r', linestyle='--', alpha=0.7)

# Add training info
plt.text(0.02, 0.85,
         f"Training Details:\n"\
         f"Sequence Length: {SEQ_LENGTH}\n"\
         f"Hidden Size: {HIDDEN_SIZE}\n"\
         f"Learning Rate: {LR}\n"\
         f"Final Loss: {losses[-1]:.6f}",
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="black", alpha=0.8),
         fontsize=12)

plt.tight_layout()

# Save the combined visualization
plt.savefig('/media/nafiz/NewVolume/ML-From-Scratch/Sequence Modeling/LSTM/lstm_complete_analysis.png', dpi=300, bbox_inches='tight')
plt.show()