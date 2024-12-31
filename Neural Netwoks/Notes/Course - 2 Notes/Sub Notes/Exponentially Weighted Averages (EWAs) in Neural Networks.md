
#### **What is EWA?**

Exponentially weighted averages (EWA) help smooth data over time, providing a more stable estimate that reduces noise. In neural networks, EWAs are often used in optimizers like Adam and RMSprop to smooth gradients, helping with convergence by balancing speed and stability.

The formula for EWA:

$$
V(t) = \beta \cdot V(t-1) + (1 - \beta) \cdot \theta(t)
$$


Where:
- ( V(t) \) is the exponentially weighted average at time step \( t \).
- ( $\beta$ ) is a smoothing factor (decay rate), typically a value close to 1 (e.g., 0.9 or 0.99).
-  $\theta(t)$  is the current value at time \( t \) (e.g., the current gradient in the case of optimizers).

#### **When to use EWA in Neural Networks?**

EWA is primarily used in optimization techniques, like in Adam, RMSprop, and other momentum-based methods. These optimizers make use of moving averages to track past gradients and reduce the variance of updates during training. This is particularly useful when:
- The gradient changes rapidly or has high variance.
- You need a more stable update rule.
- There's a need to control the speed of convergence by adjusting how much weight you put on past gradients.

#### **Why use EWA in Neural Networks?**

1. **Smooth gradient estimates**: Gradients can fluctuate, especially with noisy data. Using an exponentially weighted average can stabilize updates and lead to smoother convergence.
2. **Faster convergence**: By tracking previous gradients, the optimizer can make more informed updates and reduce the number of steps to reach an optimal solution.
3. **Prevents oscillations**: In some optimization landscapes, the gradients can cause oscillations. EWA-based methods like momentum prevent this by combining the past gradients.

#### **How to apply EWA in Neural Networks?**

In optimizers:
- **Adam Optimizer**: It uses two EWAs, one for the gradients (mean) and one for the squared gradients (variance), to normalize updates.
- **Momentum**: Momentum optimization uses EWA to "smooth" the gradient update by keeping track of the direction of the previous gradients.

### **Mathematics Behind EWA**

The formula:

$$
V(t) = \beta \cdot V(t-1) + (1 - \beta) \cdot \theta(t)
$$


This is a recursive formula that updates V(t)  based on the current value  $\theta(t)$  and the previous value \ $V(t-1)$ \.

If you expand  $V(t-1)$  recursively, you get:

$$
V(t) = \beta^t \cdot V(0) + (1 - \beta) \sum_{i=0}^{t-1} \beta^i \cdot \theta(t-i)
$$


This shows that older values \( $\theta(t-i)$ \) have exponentially less influence as time progresses due to the factor \( $\beta^i$ \), hence the name "exponentially weighted."

### **Python Implementation**

Here is a simple Python code to compute the exponentially weighted average:

```python
import numpy as np

# Function to compute EWA
def ewa(theta_values, beta=0.9):
    v_t = 0  # Initial V(0)
    v_history = []  # To store the history of EWA values

    for t, theta in enumerate(theta_values):
        v_t = beta * v_t + (1 - beta) * theta  # Recursive EWA formula
        v_history.append(v_t)

    return v_history

# Example usage: theta values could be gradients or any time-series data
theta_values = np.random.randn(100)  # Random data simulating gradients
ewa_values = ewa(theta_values, beta=0.9)

# Plotting EWA values
import matplotlib.pyplot as plt

plt.plot(theta_values, label="Theta (Original)")
plt.plot(ewa_values, label="Exponentially Weighted Average")
plt.legend()
plt.show()
```
![[EWA.png]]
# Context of NN
### **Steps to Integrate EWA:**
1. **Initialization**: Add exponentially weighted average terms for both weights and biases for each layer.
2. **Update Rule**: Apply EWA to smooth the gradients during parameter updates, similar to the momentum concept.
3. **Parameters**: Introduce a hyperparameter `beta` (decay rate for EWA).

### **Modified Code:**

```python
import numpy as np

class NN:
    def __init__(self, layer_dims, beta=0.9):
        self.layer_dims = layer_dims
        self.beta = beta  # Decay rate for EWA
        self.parameters = self.initialize_parameters()
        self.velocity = self.initialize_velocity()  # For EWA
        self.L = len(layer_dims) - 1  # Number of layers excluding input

    def initialize_parameters(self):
        np.random.seed(1)
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2. / self.layer_dims[l-1])
            parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
        return parameters

    def initialize_velocity(self):
        # Initialize the exponentially weighted average (velocity) for each layer
        velocity = {}
        for l in range(1, len(self.layer_dims)):
            velocity[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
            velocity[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])
        return velocity

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z):
        return Z > 0

    @staticmethod
    def softmax(Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_propagation(self, X):
        caches = []
        A = X
        
        for l in range(1, self.L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = np.dot(W, A_prev) + b
            
            if l == self.L:  # Output layer
                A = self.softmax(Z)
            else:
                A = self.relu(Z)
            
            cache = (A_prev, W, b, Z)
            caches.append(cache)
        return A, caches

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        # Output layer
        dZL = AL - Y
        A_prev, WL, bL, ZL = caches[self.L - 1]
        grads[f'dW{self.L}'] = (1/m) * np.dot(dZL, A_prev.T)
        grads[f'db{self.L}'] = (1/m) * np.sum(dZL, axis=1, keepdims=True)
        dA_prev = np.dot(WL.T, dZL)

        # Hidden layers
        for l in reversed(range(self.L - 1)):
            A_prev, W, b, Z = caches[l]
            dZ = dA_prev * self.relu_derivative(Z)
            grads[f'dW{l+1}'] = (1/m) * np.dot(dZ, A_prev.T)
            grads[f'db{l+1}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 0:
                dA_prev = np.dot(W.T, dZ)

        return grads

    def update_parameters(self, grads, learning_rate):
        # Update the parameters using EWA (momentum-like update)
        for l in range(1, self.L + 1):
            # Update velocity for gradients
            self.velocity[f'dW{l}'] = self.beta * self.velocity[f'dW{l}'] + (1 - self.beta) * grads[f'dW{l}']
            self.velocity[f'db{l}'] = self.beta * self.velocity[f'db{l}'] + (1 - self.beta) * grads[f'db{l}']
            
            # Update parameters using the velocity
            self.parameters[f'W{l}'] -= learning_rate * self.velocity[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * self.velocity[f'db{l}']

    def train(self, X, Y, num_iterations, learning_rate):
        costs = []
        for i in range(num_iterations):
            # Forward propagation
            AL, caches = self.forward_propagation(X)
            
            # Compute cost
            cost = -np.sum(Y * np.log(AL + 1e-8)) / Y.shape[1]
            
            # Backward propagation
            grads = self.backward_propagation(AL, Y, caches)
            
            # Update parameters using EWA
            self.update_parameters(grads, learning_rate)
            
            if i % 100 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost}")
        
        return costs

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        return np.argmax(AL, axis=0)
```

### **Key Changes:**

1. **Initialize Velocity**: 
   - A `initialize_velocity` method is introduced to store exponentially weighted averages for both weight gradients (`dW`) and bias gradients (`db`).

2. **Update Rule with EWA**: 
   - The `update_parameters` method now computes the exponentially weighted averages of the gradients before updating the parameters, using the formula:

$$
     V_dW(t) = \beta \cdot V_dW(t-1) + (1 - \beta) \cdot dW(t)
$$

 
$$
     W(t) = W(t) - \alpha \cdot V_dW(t)
$$

 Similarly for `dB` (bias updates).

### **Usage Notes:**
- The EWA concept in this context helps stabilize the gradient updates, making it smoother and less susceptible to noise during training.
- The hyperparameter `beta` controls the decay rate, typically around 0.9, but it can be tuned depending on the problem.