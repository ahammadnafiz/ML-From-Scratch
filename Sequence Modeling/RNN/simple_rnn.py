import torch
import torch.nn as nn
import torch.optim as optim

# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Initialize hidden state
        # h0 shape: (num_layers, batch_size, hidden_size)
        # Here we assume a single layer RNN
        # h0 is initialized to zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # RNN layer
        # Pass the input through the RNN layer
        # h_t = tanh(W_ih * x_t + W_hh * h_(t-1) + b_ih + b_hh)
        # out shape: (batch_size, sequence_length, hidden_size)
        out, _ = self.rnn(x, h0)
        
        # Take the last time step output
        # out shape: (batch_size, hidden_size)
        # We can use out[:, -1, :] to get the last time step output
        out = out[:, -1, :]
        
        # Output layer
        # Pass the last time step output through the fully connected layer
        # out shape: (batch_size, output_size)
        out = self.fc(out)
        
        # Sigmoid activation
        # Apply sigmoid activation to the output
        # out shape: (batch_size, 1)
        out = self.sigmoid(out)
        
        return out
    
# Example usage
def train_rnn_example():
    input_size = 10
    hidden_size = 20
    output_size = 1
    sequence_length = 5
    batch_size = 3
    
    # Create a random input tensor
    # Input shape: [batch_size, sequence_length, input_features]
    x = torch.randn(batch_size, sequence_length, input_size)
    # Create the model
    model = SimpleRNN(input_size, hidden_size, output_size)
    # Define a loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Forward pass
    for i in range(10):
        optimizer.zero_grad()
        output = model(x)
        # Create a random target tensor
        target = torch.randint(0, 2, (batch_size, output_size)).float()
        # Compute the loss
        loss = criterion(output, target)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {i+1}, Loss: {loss.item()}")
        
    # Test the model
    with torch.no_grad():
        test_output = model(x)
        predicted = (test_output > 0.5).float()
        print(f'Raw outputs: {test_output.numpy().flatten()}')
        print(f'Predictions: {predicted.numpy().flatten()}')
        
if __name__ == "__main__":
    train_rnn_example()