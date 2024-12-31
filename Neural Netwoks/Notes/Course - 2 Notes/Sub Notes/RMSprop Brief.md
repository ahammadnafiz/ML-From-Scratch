
### **1. Intuition**

When training neural networks using gradient descent, we face several issues:
- **High variance in gradients**: Gradients can vary significantly between mini-batches, leading to unstable learning.
- **Exploding gradients**: Large gradients might cause huge updates, leading to divergence in learning.
- **Vanishing gradients**: Small gradients may slow down learning, making it inefficient.

To handle these issues, we use **momentum-based methods** and **exponentially weighted averages** of the gradients. This smooths the learning process by giving recent gradients more weight but still retaining information from earlier steps. 

### **2. What is RMSprop?**

**RMSprop** stands for Root Mean Square Propagation. It maintains a running average of the squared gradients to normalize gradient updates. By squaring the gradients and taking their moving average, it helps adjust the learning rate for each parameter individually, making the learning process more robust to gradient noise.

### **3. Formula Breakdown**

Let’s examine the formulas used in your code:

1. **Initialize:**  
   - `sdW = 0`, `sdb = 0`  
   These variables hold exponentially weighted averages of the squared gradients for weights `W` and biases `b`.

2. **On iteration `t`:**
   - Compute `dW`, `db` for the current mini-batch.  
	 These are the gradients of the weights and biases based on the loss function of the current mini-batch.
   
3. **Update the exponentially weighted average for the squared gradients:**
   
$$
   \text{sdW} = \beta \cdot \text{sdW} + (1 - \beta) \cdot dW^2
$$
   
   
$$
   \text{sdb} = \beta \cdot \text{sdb} + (1 - \beta) \cdot db^2
$$
   \]
   Where:
   - `β` is a hyperparameter that controls the rate of decay (typically, β = 0.9).
   - $dW^2$ and $db^2$ are the element-wise squares of the gradients.
   
   These steps essentially take the **moving average** of the squared gradients. The current gradient has a weight of `(1 - β)`, while past gradients decay with factor `β`.

4. **Update the weights and biases:**
   
$$
   W = W - \frac{\text{learning\_rate} \cdot dW}{\sqrt{\text{sdW}} + \epsilon}
$$
   
   
$$
   b = b - \frac{\text{learning\_rate} \cdot db}{\sqrt{\text{sdb}} + \epsilon}
$$
   
   Here, `ε` is a small number (e.g., `1e-8`) added for numerical stability to avoid division by zero.
   
   These equations **normalize the gradients** using the running average of their squares, ensuring more stable updates.

### **4. Key Concepts**

- **Momentum**: Using the average of past gradients to smooth out the update step.
- **RMSprop**: Normalizing gradients using the moving average of their squared values.
- **Gradient Scaling**: Large gradients are scaled down, and small gradients are scaled up, helping with convergence.

### **5. Why and How Does It Help?**

- **Faster Convergence**: Since the learning rate is adjusted for each parameter individually, training can converge faster by taking larger steps where possible and smaller steps where necessary.
- **Better Handling of Vanishing/Exploding Gradients**: The normalization prevents very large or very small gradient updates, avoiding issues like vanishing or exploding gradients in deep networks.
- **Smoothing of Updates**: Instead of using the immediate gradient (which can fluctuate a lot), we use a smoothed version. This makes learning more stable and less noisy, especially in mini-batch gradient descent.

### **6. Example of Gradient Descent with RMSprop**

Consider a simple case of updating a weight `W` in one dimension (for clarity).

1. Initial values:
   - `W = 1.0`
   - `learning_rate = 0.01`
   - `beta = 0.9`

2. Suppose the gradients `dW` in different iterations are:
   - Iteration 1: `dW = 0.8`
   - Iteration 2: `dW = 0.9`
   - Iteration 3: `dW = 0.1`
   - Iteration 4: `dW = -0.2`

   The squared gradients would be:
   - Iteration 1: `dW^2 = 0.64`
   - Iteration 2: `dW^2 = 0.81`
   - Iteration 3: `dW^2 = 0.01`
   - Iteration 4: `dW^2 = 0.04`

3. Applying the update:
   After each iteration, you update the smoothed squared gradients and the weight `W`.

   After a few iterations, you'll see that:
   - **Larger gradients** will be scaled down (because the squared value grows).
   - **Smaller gradients** will be scaled up (because the squared value remains small).

This dynamic adjustment allows the learning process to adapt to the gradient magnitude, speeding up learning in directions where the gradients are small and slowing down learning where the gradients are large.

---

### **7. When Should You Use RMSprop?**

- **Deep Neural Networks**: It’s especially useful in very deep networks where vanishing and exploding gradients are common.
- **Mini-batch Gradient Descent**: When training on mini-batches, where gradient noise is higher, RMSprop helps smooth out these fluctuations.
- **Non-stationary Problems**: Problems where the data distribution changes over time benefit from the adaptability of RMSprop.

---

### **8. Implementation in the Neural Network Code**


```python
import numpy as np

class NN:
    def __init__(self, layer_dims, beta=0.9, learning_rate=0.01):
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters()
        self.sdW = {f'sdW{l}': np.zeros_like(self.parameters[f'W{l}']) for l in range(1, len(self.layer_dims))}
        self.sdb = {f'sdb{l}': np.zeros_like(self.parameters[f'b{l}']) for l in range(1, len(self.layer_dims))}
        self.beta = beta
        self.learning_rate = learning_rate
        self.L = len(layer_dims) - 1  # number of layers excluding input

    def initialize_parameters(self):
        np.random.seed(1)
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2. / self.layer_dims[l-1])
            parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
        return parameters

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

    def update_parameters(self, grads):
        epsilon = 1e-8  # Small value to avoid division by zero

        for l in range(1, self.L + 1):
            # Update the exponentially weighted averages of the squared gradients
            self.sdW[f'sdW{l}'] = (self.beta * self.sdW[f'sdW{l}']) + (1 - self.beta) * np.square(grads[f'dW{l}'])
            self.sdb[f'sdb{l}'] = (self.beta * self.sdb[f'sdb{l}']) + (1 - self.beta) * np.square(grads[f'db{l}'])

            # Update weights and biases using RMSprop-like update rule
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}'] / (np.sqrt(self.sdW[f'sdW{l}']) + epsilon)
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}'] / (np.sqrt(self.sdb[f'sdb{l}']) + epsilon)

    def train(self, X, Y, num_iterations):
        costs = []
        for i in range(num_iterations):
            # Forward propagation
            AL, caches = self.forward_propagation(X)
            
            # Compute cost
            cost = -np.sum(Y * np.log(AL + 1e-8)) / Y.shape[1]
            
            # Backward propagation
            grads = self.backward_propagation(AL, Y, caches)
            
            # Update parameters with RMSprop
            self.update_parameters(grads)
            
            if i % 100 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost}")
        
        return costs

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        return np.argmax(AL, axis=0)

# Example usage
if __name__ == "__main__":
    X = np.random.randn(3, 100)  # 3 input features, 100 samples
    Y = np.eye(2)[np.random.randint(0, 2, 100)].T  # 2 output classes, one-hot encoded

    layer_dims = [3, 5, 2]  # 3 input features, 5 neurons in hidden layer, 2 output classes
    model = NN(layer_dims, beta=0.9, learning_rate=0.01)
    model.train(X, Y, num_iterations=1000)

```
