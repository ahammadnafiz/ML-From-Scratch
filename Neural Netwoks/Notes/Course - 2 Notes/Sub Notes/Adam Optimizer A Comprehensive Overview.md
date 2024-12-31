#### **What is Adam?**
Adam (short for Adaptive Moment Estimation) is an advanced optimization algorithm used for training deep learning models. It combines the benefits of two other popular optimization algorithms: **Momentum** and **RMSprop**.

- **Momentum** helps accelerate gradient descent by considering the exponentially weighted average of past gradients.
- **RMSprop** adjusts the learning rate for each parameter individually by using the squared gradients, effectively slowing down updates for high variance parameters and speeding up updates for low variance ones.

Adam merges these two concepts and adds bias correction, which ensures more accurate estimations during the early stages of training.

#### **How Adam Works (Mathematically)**

Given gradients \( dW \) (for weights) and \( db \) (for biases) at each step of training:

1. **Exponential Moving Average of the Gradient (Momentum)**:
   
$$
   v_{dw} = \beta_1 \cdot v_{dw} + (1 - \beta_1) \cdot dW
$$
   
   
$$
   v_{db} = \beta_1 \cdot v_{db} + (1 - \beta_1) \cdot db
$$
   
   Where:
   -  $\beta_1$  is the decay rate for momentum (commonly set to 0.9).
   - ( $v_{dw}$ \) and \( $v_{db}$ \) are exponentially weighted averages of the gradients.

2. **Exponential Moving Average of the Squared Gradient (RMSprop)**:
   
$$
   s_{dw} = \beta_2 \cdot s_{dw} + (1 - \beta_2) \cdot (dW)^2
$$
   
   
$$
   s_{db} = \beta_2 \cdot s_{db} + (1 - \beta_2) \cdot (db)^2
$$
   
   Where:
   -  $\beta_2$  is the decay rate for the squared gradients (typically set to 0.999).
   - ( $s_{dw}$ \) and \( $s_{db}$ \) are exponentially weighted averages of squared gradients.

3. **Bias Correction**:
   Since both the momentum and RMSprop estimations are biased toward zero at the start of training, we apply bias correction:
   
$$
   \hat{v}_{dw} = \frac{v_{dw}}{1 - \beta_1^t}
$$
   
   
$$
   \hat{v}_{db} = \frac{v_{db}}{1 - \beta_1^t}
$$
   
   
$$
   \hat{s}_{dw} = \frac{s_{dw}}{1 - \beta_2^t}
$$
   
   
$$
   \hat{s}_{db} = \frac{s_{db}}{1 - \beta_2^t}
$$
   
   Where \( t \) is the iteration number.

4. **Parameter Updates**:
   Finally, the parameters are updated using the corrected estimates:
   
$$
   W = W - \alpha \cdot \frac{\hat{v}_{dw}}{\sqrt{\hat{s}_{dw}} + \epsilon}
$$
   
   
$$
   b = b - \alpha \cdot \frac{\hat{v}_{db}}{\sqrt{\hat{s}_{db}} + \epsilon}
$$
   
   Where:
   -  $\alpha$  is the learning rate.
   -  $\epsilon$  is a small constant (typically \( $10^{-8}$ \)) to avoid division by zero.

#### **Why Adam?**
1. **Faster Convergence**: The combination of momentum and adaptive learning rates allows Adam to converge faster and more efficiently than other optimizers like standard SGD.
2. **Robustness**: Adam is particularly useful for noisy data, sparse gradients, or data with a large number of features.
3. **Less Tuning**: Adam performs well with minimal hyperparameter tuning, using default settings of \( $\beta_1 = 0.9$ \), \( $\beta_2 = 0.999$ \), and \( $\epsilon = 10^{-8}$ \).

#### **When to Use Adam?**
- **When training deep neural networks**: Especially those with high variance in gradients or sparse gradients.
- **When fast convergence is important**: Adam's adaptive learning rates help optimize the process without excessive fine-tuning.
- **When using large datasets**: Adam can handle larger datasets more efficiently than standard SGD due to its adaptive properties.

### **Adam vs. Other Optimizers**
- **SGD**: Slower, but may find better minima in some cases.
- **Momentum**: Adds speed to gradient descent but doesn't adapt the learning rate.
- **RMSprop**: Adapts learning rates per parameter but lacks momentum.
- **Adam**: Combines both momentum and adaptive learning rates, leading to more efficient training.

### **Code Implementation of Adam in Your `NN` Class**

Here is how you can implement the Adam optimizer in your `NN` class:

```python
import numpy as np

class NN:
    def __init__(self, layer_dims, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters()
        self.vdW = {f'vdW{l}': np.zeros_like(self.parameters[f'W{l}']) for l in range(1, len(self.layer_dims))}
        self.vdb = {f'vdb{l}': np.zeros_like(self.parameters[f'b{l}']) for l in range(1, len(self.layer_dims))}
        self.sdW = {f'sdW{l}': np.zeros_like(self.parameters[f'W{l}']) for l in range(1, len(self.layer_dims))}
        self.sdb = {f'sdb{l}': np.zeros_like(self.parameters[f'b{l}']) for l in range(1, len(self.layer_dims))}
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Iteration count
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

    def update_parameters_adam(self, grads):
        self.t += 1
        for l in range(1, self.L + 1):
            # Update momentum
            self.vdW[f'vdW{l}'] = (self.beta1 * self.vdW[f'vdW{l}']) + (1 - self.beta1) * grads[f'dW{l}']
            self.vdb[f'vdb{l}'] = (self.beta1 * self.vdb[f'vdb{l}']) + (1 -

 self.beta1) * grads[f'db{l}']

            # Update RMSprop
            self.sdW[f'sdW{l}'] = (self.beta2 * self.sdW[f'sdW{l}']) + (1 - self.beta2) * np.square(grads[f'dW{l}'])
            self.sdb[f'sdb{l}'] = (self.beta2 * self.sdb[f'sdb{l}']) + (1 - self.beta2) * np.square(grads[f'db{l}'])

            # Bias correction
            vdW_corrected = self.vdW[f'vdW{l}'] / (1 - np.power(self.beta1, self.t))
            vdb_corrected = self.vdb[f'vdb{l}'] / (1 - np.power(self.beta1, self.t))
            sdW_corrected = self.sdW[f'sdW{l}'] / (1 - np.power(self.beta2, self.t))
            sdb_corrected = self.sdb[f'sdb{l}'] / (1 - np.power(self.beta2, self.t))

            # Update parameters
            self.parameters[f'W{l}'] -= self.learning_rate * vdW_corrected / (np.sqrt(sdW_corrected) + self.epsilon)
            self.parameters[f'b{l}'] -= self.learning_rate * vdb_corrected / (np.sqrt(sdb_corrected) + self.epsilon)

    def train(self, X, Y, iterations=1000):
        for i in range(iterations):
            AL, caches = self.forward_propagation(X)
            grads = self.backward_propagation(AL, Y, caches)
            self.update_parameters_adam(grads)
            if i % 100 == 0:
                cost = -np.sum(Y * np.log(AL)) / X.shape[1]
                print(f"Iteration {i}: Cost = {cost}")
```

### **Explanation of Code:**
1. **Parameter Initialization**: We initialize the parameters for weights and biases and create zero-filled arrays for `vdW`, `vdb`, `sdW`, and `sdb` to store the momentum and squared gradients.
2. **Forward Propagation**: We compute the activations for each layer using ReLU for hidden layers and softmax for the output layer.
3. **Backward Propagation**: We compute gradients using the chain rule and store them.
4. **Adam Update**: Using the Adam formula, the parameters are updated with bias correction and momentum.
5. **Training Loop**: The model trains for a fixed number of iterations, updating parameters and printing the cost every 100 iterations.