# Pure GRU Implementation from Scratch
# Demonstrates the core mechanics of a Gated Recurrent Unit

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    """Tanh activation function"""
    return np.tanh(np.clip(x, -500, 500))

def dtanh(x):
    """Derivative of tanh"""
    return 1 - np.tanh(x)**2

def dsigmoid(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

class SimpleGRU:
    """
    A simple GRU cell implementation showing the core mechanics
    
    GRU has three main components:
    1. Reset Gate (r): Controls how much of previous hidden state to forget
    2. Update Gate (z): Controls how much of new vs old information to keep
    3. Candidate Hidden State (h_hat): New information to potentially add
    """
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights with small random values
        scale = 0.1
        
        # Reset gate weights: W_r * x + U_r * h + b_r
        self.W_r = np.random.randn(input_size, hidden_size) * scale
        self.U_r = np.random.randn(hidden_size, hidden_size) * scale
        self.b_r = np.zeros((1, hidden_size))
        
        # Update gate weights: W_z * x + U_z * h + b_z
        self.W_z = np.random.randn(input_size, hidden_size) * scale
        self.U_z = np.random.randn(hidden_size, hidden_size) * scale
        self.b_z = np.zeros((1, hidden_size))
        
        # Candidate hidden state weights: W * x + U * (r ⊙ h) + b
        self.W = np.random.randn(input_size, hidden_size) * scale
        self.U = np.random.randn(hidden_size, hidden_size) * scale
        self.b = np.zeros((1, hidden_size))
        
        # Store intermediate values for analysis
        self.last_gates = {}
        
    def forward(self, x, h_prev):
        """
        Forward pass through GRU cell
        
        Args:
            x: Input at current time step (batch_size, input_size)
            h_prev: Hidden state from previous time step (batch_size, hidden_size)
            
        Returns:
            h: New hidden state (batch_size, hidden_size)
        """
        
        # 1. Reset Gate: Decides what parts of previous hidden state to forget
        r = sigmoid(x @ self.W_r + h_prev @ self.U_r + self.b_r)
        
        # 2. Update Gate: Decides how much new vs old information to keep
        z = sigmoid(x @ self.W_z + h_prev @ self.U_z + self.b_z)
        
        # 3. Candidate Hidden State: New information we might want to add
        # Note: r ⊙ h_prev means element-wise multiplication (Hadamard product)
        h_hat = tanh(x @ self.W + (r * h_prev) @ self.U + self.b)
        
        # 4. Final Hidden State: Interpolate between old and new information
        # h = (1 - z) ⊙ h_prev + z ⊙ h_hat
        h = (1 - z) * h_prev + z * h_hat
        
        # Store for analysis
        self.last_gates = {
            'reset': r,
            'update': z,
            'candidate': h_hat,
            'output': h,
            'prev_hidden': h_prev
        }
        
        return h
    
    def get_gate_info(self):
        """Return information about the last forward pass for analysis"""
        return self.last_gates.copy()

def create_simple_sequence():
    """Create a simple test sequence"""
    # Create a sequence that goes: [1, 0, 1, 0, 1]
    # This will help us see how GRU maintains memory
    sequence = []
    for i in range(5):
        if i % 2 == 0:
            sequence.append(np.array([[1.0, 0.0]]))  # Input: [1, 0]
        else:
            sequence.append(np.array([[0.0, 1.0]]))  # Input: [0, 1]
    return sequence

def run_gru_demo():
    """Demonstrate GRU mechanics step by step"""
    
    print("=" * 60)
    print("GRU MECHANICS DEMONSTRATION")
    print("=" * 60)
    
    # Initialize GRU
    input_size = 2
    hidden_size = 3
    gru = SimpleGRU(input_size, hidden_size)
    
    # Create test sequence
    sequence = create_simple_sequence()
    
    print(f"\nGRU Configuration:")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Sequence length: {len(sequence)}")
    
    # Initialize hidden state
    h = np.zeros((1, hidden_size))
    
    print(f"\nInitial hidden state: {h.flatten()}")
    print("\n" + "-" * 60)
    
    # Process sequence step by step
    hidden_states = [h.copy()]
    
    for t, x in enumerate(sequence):
        print(f"\nStep {t + 1}:")
        print(f"Input: {x.flatten()}")
        print(f"Previous hidden: {h.flatten()}")
        
        # Forward pass
        h = gru.forward(x, h)
        hidden_states.append(h.copy())
        
        # Get gate information
        gates = gru.get_gate_info()
        
        print(f"\nGate Analysis:")
        print(f"  Reset gate (r):     {gates['reset'].flatten()}")
        print(f"  Update gate (z):    {gates['update'].flatten()}")
        print(f"  Candidate (h_hat):  {gates['candidate'].flatten()}")
        print(f"  New hidden (h):     {gates['output'].flatten()}")
        
        # Explain what's happening
        r_mean = np.mean(gates['reset'])
        z_mean = np.mean(gates['update'])
        
        print(f"\nInterpretation:")
        if r_mean < 0.3:
            print(f"  - Reset gate is LOW ({r_mean:.3f}) → Forgetting most of previous state")
        elif r_mean > 0.7:
            print(f"  - Reset gate is HIGH ({r_mean:.3f}) → Keeping most of previous state")
        else:
            print(f"  - Reset gate is MEDIUM ({r_mean:.3f}) → Partial memory retention")
            
        if z_mean < 0.3:
            print(f"  - Update gate is LOW ({z_mean:.3f}) → Keeping mostly old information")
        elif z_mean > 0.7:
            print(f"  - Update gate is HIGH ({z_mean:.3f}) → Using mostly new information")
        else:
            print(f"  - Update gate is MEDIUM ({z_mean:.3f}) → Mixing old and new information")
        
        print("-" * 60)
    
    return hidden_states

def analyze_memory_retention():
    """Analyze how well GRU retains information over time"""
    
    print("\n" + "=" * 60)
    print("MEMORY RETENTION ANALYSIS")
    print("=" * 60)
    
    input_size = 1
    hidden_size = 4
    gru = SimpleGRU(input_size, hidden_size)
    
    # Create a sequence with a strong signal at the beginning, then noise
    sequence_length = 10
    sequence = []
    
    # Strong signal at the beginning
    sequence.append(np.array([[1.0]]))  # Important information
    
    # Followed by weak noise
    for i in range(sequence_length - 1):
        sequence.append(np.array([[0.1 * np.sin(i)]]))  # Weak noise
    
    print(f"Testing memory retention over {sequence_length} steps")
    print("First input is strong signal (1.0), rest is weak noise")
    
    h = np.zeros((1, hidden_size))
    memory_strength = []
    
    for t, x in enumerate(sequence):
        h = gru.forward(x, h)
        gates = gru.get_gate_info()
        
        # Measure how much of the hidden state correlates with initial signal
        memory_strength.append(np.mean(np.abs(h)))
        
        if t == 0:
            initial_state = h.copy()
            print(f"Step {t}: Input={x.flatten()[0]:.3f}, Hidden strength={memory_strength[-1]:.3f}")
        elif t % 3 == 0 or t == len(sequence) - 1:
            correlation = np.corrcoef(initial_state.flatten(), h.flatten())[0, 1]
            print(f"Step {t}: Input={x.flatten()[0]:.3f}, Hidden strength={memory_strength[-1]:.3f}, Correlation with initial={correlation:.3f}")
    
    return memory_strength

def show_gru_equations():
    """Display the mathematical equations behind GRU"""
    
    print("\n" + "=" * 60)
    print("GRU MATHEMATICAL EQUATIONS")
    print("=" * 60)
    
    equations = """
The GRU cell computes the following at each time step t:

1. Reset Gate:
   r_t = σ(W_r · x_t + U_r · h_{t-1} + b_r)
   └─ Controls how much of previous hidden state to "forget"

2. Update Gate:
   z_t = σ(W_z · x_t + U_z · h_{t-1} + b_z)
   └─ Controls how much to update vs keep from previous state

3. Candidate Hidden State:
   ĥ_t = tanh(W · x_t + U · (r_t ⊙ h_{t-1}) + b)
   └─ New information we might want to add
   └─ Note: r_t ⊙ h_{t-1} is element-wise multiplication

4. Final Hidden State:
   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ĥ_t
   └─ Interpolate between old (h_{t-1}) and new (ĥ_t) information

Where:
- σ is the sigmoid function: σ(x) = 1/(1 + e^(-x))
- tanh is the hyperbolic tangent function
- ⊙ denotes element-wise multiplication (Hadamard product)
- W, U are weight matrices, b are bias vectors
- x_t is input at time t, h_t is hidden state at time t

Key Insights:
- When z_t ≈ 0: Keep old information (h_t ≈ h_{t-1})
- When z_t ≈ 1: Use new information (h_t ≈ ĥ_t)
- When r_t ≈ 0: Ignore previous state when computing candidate
- When r_t ≈ 1: Fully consider previous state for candidate
"""
    
    print(equations)

if __name__ == "__main__":
    # Show the mathematical foundation
    show_gru_equations()
    
    # Run the main demonstration
    hidden_states = run_gru_demo()
    
    # Analyze memory retention
    memory_strength = analyze_memory_retention()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
This demonstration shows the core mechanics of a GRU:

1. GATES: The reset and update gates control information flow
2. MEMORY: GRU can selectively remember or forget information
3. DYNAMICS: The hidden state evolves based on input and gates
4. FLEXIBILITY: Gates adapt to learn what information is important

Key advantages of GRU over simple RNN:
- Mitigates vanishing gradient problem
- Can learn long-term dependencies
- Computationally efficient (fewer parameters than LSTM)
- Good balance between complexity and performance
""")