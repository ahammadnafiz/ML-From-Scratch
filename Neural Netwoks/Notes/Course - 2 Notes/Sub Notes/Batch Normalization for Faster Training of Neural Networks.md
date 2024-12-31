
In deep learning, **batch normalization** is a technique used to normalize the inputs to a neural network layer in order to stabilize and speed up the training process. The key question here is: _Can we normalize the activations of a hidden layer \( A[l] \) to train \( W[l+1] \) and \( b[l+1] \) faster?_ 

The answer is yes, but with important distinctions. While it is technically possible to normalize \( A[l] \) (the post-activation outputs), **in practice, normalizing \( Z[l] \)** (the pre-activation inputs) is much more common and yields better results. This is the basis of **batch normalization**, which was popularized by researchers such as Sergey Ioffe and Christian Szegedy, and is also endorsed by Andrew Ng in his deep learning courses.

#### When to Use Batch Normalization:
- **During training**: Batch normalization is applied after computing \( Z[l] \) (before the activation function) to ensure that the inputs to the next layer have a mean of 0 and variance of 1, which helps accelerate training.
- **Deep networks**: Batch normalization is particularly effective in deep neural networks, where vanishing/exploding gradients are common. It also allows for higher learning rates without risking divergence.

#### Why Use Batch Normalization:
1. **Improves gradient flow**: Normalizing the activations helps mitigate issues with vanishing and exploding gradients, which can hinder learning in deep networks.
2. **Speeds up convergence**: By ensuring consistent distributions of inputs, batch normalization allows the network to converge faster.
3. **Reduces internal covariate shift**: By stabilizing the distribution of inputs across layers, it helps the network learn more efficiently.
4. **Acts as a regularizer**: Batch normalization has a slight regularization effect, which can reduce the need for dropout.

#### How Batch Normalization Works:
For a given layer \( l \), the output before applying the activation function is \( Z[l] \). The batch normalization algorithm follows these steps:

1. **Compute the mean**:  
   $\mu = \frac{1}{m} \sum_{i=1}^{m} z_i$ 
   
   This is the mean of the pre-activation values \( Z[l] \), where \( m \) is the batch size.
   
2. **Compute the variance**:  
    $\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (z_i - \mu)^2$ 

3. **Normalize \( Z[l] \)**:  
   $Z_{\text{norm}, i} = \frac{z_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$ 
   
   Here, \( \epsilon \) is a small constant added for numerical stability, especially when the variance is close to 0.

4. **Scale and shift** (learnable parameters):  
	$Z_{\tilde{i}} = \gamma \cdot Z_{\text{norm}, i} + \beta$
   
   - ( $\gamma$ \) ($scale$) and \( $\beta$ \) ($shift$) are learnable parameters that allow the network to learn different distributions for the normalized inputs.
   - If \( $\gamma = \sqrt{\sigma^2 + \epsilon}$ \) and \( $\beta = \mu$ \), then the original \( Z \) is recovered.

5. **Pass through activation function**: After normalizing and scaling \( Z[l] \), the values are passed through an activation function (e.g., ReLU) to produce \( A[l] \).

#### Practical Examples

```python
import numpy as np

class BatchNormLayer:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
    
    def initialize_params(self, Z):
        self.gamma = np.ones(Z.shape[1])  # Initialize gamma to ones
        self.beta = np.zeros(Z.shape[1])  # Initialize beta to zeros

    def forward(self, Z):
        """
        Forward pass of Batch Normalization.
        Args:
        - Z: pre-activation values (batch of inputs to normalize)
        """
        # Compute mean and variance for each feature in the batch
        mean = np.mean(Z, axis=0)
        variance = np.var(Z, axis=0)

        # Normalize
        Z_norm = (Z - mean) / np.sqrt(variance + self.epsilon)

        # Scale and shift
        Z_tilde = self.gamma * Z_norm + self.beta
        
        return Z_tilde

    def update_params(self, dgamma, dbeta, learning_rate):
        """
        Update learnable parameters gamma and beta
        Args:
        - dgamma: gradient of loss with respect to gamma
        - dbeta: gradient of loss with respect to beta
        - learning_rate: step size for updating parameters
        """
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta

# Example usage:
Z = np.random.randn(5, 3)  # A batch of 5 inputs, each with 3 features
batch_norm = BatchNormLayer()
batch_norm.initialize_params(Z)

# Forward pass
Z_tilde = batch_norm.forward(Z)
print("Normalized and scaled Z:", Z_tilde)
```

### Key Takeaways:
- **Normalization of \( Z[l] \) is common practice** because it ensures stability during training and helps with faster convergence. Normalizing \( A[l] \) (post-activation outputs) is less common and can disrupt the optimization process.
- **Batch normalization** introduces learnable parameters \( \gamma \) (scale) and \( \beta \) (shift) to allow the network to learn different distributions.
- The main advantage of batch normalization is the ability to use **higher learning rates** without the risk of divergence, leading to **faster training** and **better generalization**.

# Intuition Behind Backpropagation with Batch Normalization

Batch normalization (BN) helps stabilize and speed up training by normalizing the activations (intermediate values like \( Z \)) in each layer to have zero mean and unit variance. When we incorporate batch normalization into a neural network, we introduce additional parameters (scale \( \gamma \) and shift \( \beta \)) and also modify the way we propagate gradients through the network during backpropagation.

In backpropagation, the goal is to compute how much each parameter (weights \( W \), biases \( b \), batch norm parameters \( \gamma \), and \( \beta \)) should change to reduce the loss. When batch normalization is introduced, it adds complexity to this gradient computation because we need to account for how the normalization affects the network’s behavior during training.

### Batch Normalization: Forward Pass Recap

In the forward pass for a layer \( l \), we typically calculate:

1. **Affine transformation** (linear part of a neural network):
   
$$
   Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$$
   
   where:
   -  $W^{[l]}$  and  $b^{[l]}$  are the weights and biases for layer \( l \),
   - $A^{[l-1]}$  is the activation from the previous layer.

2. **Batch normalization**: Instead of immediately applying the activation function on \( $Z^{[l]}$ \), we first normalize it.
   
$$
   \mu^{[l]} = \frac{1}{m} \sum_{i=1}^{m} Z_i^{[l]} \quad \text{(mean)}
$$
   
   
$$
   \sigma^{2[l]} = \frac{1}{m} \sum_{i=1}^{m} (Z_i^{[l]} - \mu^{[l]})^2 \quad \text{(variance)}
$$
   
   
$$
   Z_{\text{norm}}^{[l]} = \frac{Z^{[l]} - \mu^{[l]}}{\sqrt{\sigma^{2[l]} + \epsilon}} \quad \text{(normalize to zero mean and unit variance)}
$$
   
   
3. **Scale and shift**:
   
$$
   Z_{\text{tilde}}^{[l]} = \gamma^{[l]} Z_{\text{norm}}^{[l]} + \beta^{[l]}
$$
   
   where:
   -  $\gamma^{[l]}$ and  $\beta^{[l]}$  are the learnable parameters that allow the network to adjust the scale and shift of the normalized values.

4. **Activation function** (e.g., ReLU):
   
$$
   A^{[l]} = g(Z_{\text{tilde}}^{[l]})
$$
   

### Backpropagation Through Batch Normalization: Mathematical Details

When performing backpropagation, we need to compute how the gradients of the loss \( L \) propagate back through the batch normalization process. Let's break down the steps:

#### 1. Gradient of the Loss with Respect to \( Z_{\text{tilde}}^{[l]} \) (output of batch normalization):
   - This is the usual gradient from backpropagation:
 
$$
     \frac{\partial L}{\partial Z_{\text{tilde}}^{[l]}} = \frac{\partial L}{\partial A^{[l]}} \cdot g'(Z_{\text{tilde}}^{[l]})
$$
 
 where \( $g'(Z_{\text{tilde}}^{[l]})$ \) is the derivative of the activation function (e.g., ReLU).

#### 2. Gradients of the Loss with Respect to \( $\gamma^{[l]}$ \) and \( $\beta^{[l]}$ \):
   - These gradients are relatively simple because they are directly involved in the linear transformation after normalization:
     \[
$$
     \frac{\partial L}{\partial \gamma^{[l]}} = \sum_{i=1}^{m} \frac{\partial L}{\partial Z_{\text{tilde}, i}^{[l]}} \cdot Z_{\text{norm}, i}^{[l]}
$$
 
 
$$
     \frac{\partial L}{\partial \beta^{[l]}} = \sum_{i=1}^{m} \frac{\partial L}{\partial Z_{\text{tilde}, i}^{[l]}}
$$
 

#### 3. Gradient of the Loss with Respect to \( Z_{\text{norm}}^{[l]} \):
   - The normalized values \( $Z_{\text{norm}}^{[l]}$ \) are scaled by \( $\gamma^{[l]}$ \), so we need to account for that when propagating the gradient back:
 
$$
     \frac{\partial L}{\partial Z_{\text{norm}}^{[l]}} = \frac{\partial L}{\partial Z_{\text{tilde}}^{[l]}} \cdot \gamma^{[l]}
$$
 

#### 4. Gradient of the Loss with Respect to \( $\mu^{[l]}$ \) and \( $\sigma^{2[l]}$ \) (mean and variance):
   To compute these gradients, we use the chain rule on the normalization step. The gradient of the loss with respect to \( $Z^{[l]}$ \) needs to account for both the shift (mean subtraction) and scaling (variance division).

   **Gradient with respect to the variance** \( $\sigma^{2[l]}$ \):
   
$$
   \frac{\partial L}{\partial \sigma^{2[l]}} = \sum_{i=1}^{m} \frac{\partial L}{\partial Z_{\text{norm}, i}^{[l]}} \cdot (Z_i^{[l]} - \mu^{[l]}) \cdot \left( -\frac{1}{2} \right) \cdot (\sigma^{2[l]} + \epsilon)^{-\frac{3}{2}}
$$
   

   **Gradient with respect to the mean** \( $\mu^{[l]}$ \):
   
$$
   \frac{\partial L}{\partial \mu^{[l]}} = \sum_{i=1}^{m} \frac{\partial L}{\partial Z_{\text{norm}, i}^{[l]}} \cdot \left( - \frac{1}{\sqrt{\sigma^{2[l]} + \epsilon}} \right) + \frac{\partial L}{\partial \sigma^{2[l]}} \cdot \frac{-2}{m} \sum_{i=1}^{m} (Z_i^{[l]} - \mu^{[l]})
$$
   

#### 5. Gradient of the Loss with Respect to \( $Z^{[l]}$ \) (original pre-activation values):
   Finally, using the gradients with respect to \( $\mu^{[l]}$ \) and \( $\sigma^{2[l]}$ \), we compute the gradient with respect to the original \( $Z^{[l]}$ \):

   
$$
   \frac{\partial L}{\partial Z_i^{[l]}} = \frac{\partial L}{\partial Z_{\text{norm}, i}^{[l]}} \cdot \frac{1}{\sqrt{\sigma^{2[l]} + \epsilon}} + \frac{\partial L}{\partial \sigma^{2[l]}} \cdot \frac{2(Z_i^{[l]} - \mu^{[l]})}{m} + \frac{\partial L}{\partial \mu^{[l]}} \cdot \frac{1}{m}
$$
   

This equation propagates the gradients back to the original inputs to the batch normalization layer.

### Summary of Backpropagation Through Batch Normalization:
1. Compute the gradient with respect to the outputs \( $Z_{\text{tilde}}$ \) of the batch normalization layer.
2. Compute the gradients for the learnable parameters \( $\gamma$ \) and \( $\beta$ \).
3. Propagate the gradient back through the normalization step to get the gradients for the mean and variance.
4. Use these to compute the gradient with respect to the original input \( Z \) to the batch normalization layer.

This formula corresponds to the **backpropagation through batch normalization** process. The objective is to calculate the gradients for the input \( Z \) (the input to the batch normalization layer), as well as the learnable parameters \( \gamma \) (scale) and \( \beta \) (shift) used during the normalization. Below is an explanation of each part and how they derive from the original batch normalization backpropagation theory.

### Formula Breakdown

#### 1. $m = Z.shape[1]$ 

This defines the size of the batch. Specifically, \( Z \) is the output of a layer before batch normalization is applied, and `m` refers to the number of samples in a batch (the second dimension of \( Z \)).

#### 2. \( $Z_{\mu} = Z - \text{mean}$ \)

This step centers \( Z \) by subtracting the mean of \( Z \). The mean subtraction is the first step in batch normalization, ensuring that the data has zero mean. In other words:


$$
Z_{\mu} = Z - \mu
$$

where \( $\mu$ \) is the batch mean.

#### 3.  $\text{std\_inv} = \frac{1}{\sqrt{\text{variance} + \epsilon}}$ 

This calculates the inverse of the standard deviation. The standard deviation is part of the normalization process, which scales \( Z \) so that it has unit variance. The formula includes a small constant \( $\epsilon$ \) to prevent division by zero (for numerical stability).


$$
\text{std\_inv} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}
$$

where \( $\sigma^2$ \) is the variance of the batch.

#### 4. \( $dZ$ \)

This is the gradient with respect to the input \( Z \), the core result of the backpropagation process. The formula for \( dZ \) is derived using the chain rule, accounting for the normalization process:


$$
dZ = \left( \frac{1}{m} \right) \cdot \gamma \cdot \text{std\_inv} \cdot \left( m \cdot dZ_{\text{norm}} - \sum dZ_{\text{norm}} - Z_{\mu} \cdot \text{std\_inv}^2 \cdot \sum (dZ_{\text{norm}} \cdot Z_{\mu}) \right)
$$


Let’s break this down:

-  $dZ_{\text{norm}}$ is the gradient of the loss with respect to the normalized \( Z \) (i.e., \( $Z_{\text{norm}}$ \)). 
- $\gamma \cdot \text{std\_inv}$  accounts for the scaling and shifting done by the batch normalization.
- The expression inside the parentheses \( $\left( m \cdot dZ_{\text{norm}} - \sum dZ_{\text{norm}} - Z_{\mu} \cdot \text{std\_inv}^2 \cdot \sum (dZ_{\text{norm}} \cdot Z_{\mu}) \right)$ \) represents the backpropagation of the gradients through the mean and variance steps.

  -  $m \cdot dZ_{\text{norm}}$: The first term scales the gradient across the batch size.
  -  $\sum dZ_{\text{norm}}$ : The second term subtracts the sum of all gradients, ensuring the mean constraint.
  - $Z_{\mu} \cdot \text{std\_inv}^2 \cdot \sum (dZ_{\text{norm}} \cdot Z_{\mu})$ : The third term adjusts for the variance, ensuring gradients correctly propagate through the standard deviation.

#### 5. \( $d\gamma = \sum dZ_{\text{norm}} \cdot Z_{\mu} \cdot \text{std\_inv}$ \)

This is the gradient with respect to the scale parameter \( $\gamma$ \). It accumulates the gradient of the loss with respect to the scaled normalized input \( $Z_{\text{norm}}$ \), weighted by how much the original \( Z \) was normalized.


$$
d\gamma = \sum dZ_{\text{norm}} \cdot Z_{\mu} \cdot \text{std\_inv}
$$


This gradient indicates how the scaling factor \( \gamma \) should change based on the contribution of each normalized input to the loss.

#### 6. \( $d\beta = \sum dZ_{\text{norm}}$ \)

This is the gradient with respect to the shift parameter \( \beta \). Since \( \beta \) is simply added to the normalized values, its gradient is the sum of the gradients of the loss with respect to the normalized inputs.


$$
d\beta = \sum dZ_{\text{norm}}
$$


### How These Formulas Derive from Backpropagation Through Batch Normalization

To derive these expressions, we apply the **chain rule** of calculus step-by-step through each part of the batch normalization process:

1. **Start with the output** \( $Z_{\text{tilde}} = \gamma \cdot Z_{\text{norm}} + \beta$ \).
   - The gradient of the loss with respect to \( $Z_{\text{tilde}}$ \) is provided as \( $dZ_{\text{tilde}}$ \), which we use as \( $dZ_{\text{norm}}$ \).
   
2. **Backpropagate through the scaling and shifting** using \( $\gamma$ \) and \( $\beta$ \):
   -  $d\gamma = \sum dZ_{\text{norm}} \cdot Z_{\text{norm}}$ 
   -  $d\beta = \sum dZ_{\text{norm}}$ \

3. **Backpropagate through the normalization** to obtain the gradient with respect to the pre-normalized activations \( Z \):
   - Apply the chain rule to propagate through the mean and variance calculations.

4. **Combine the gradients**:
   - Account for the scaling effect of \( $\gamma$ \) and the adjustment made by the mean and variance terms, ensuring that the batch normalization properties (zero mean, unit variance) are correctly reflected in the backpropagation.

![[batch_norm_detailed.png]]
# Code

```python
import numpy as np

class NN:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters()
        self.L = len(layer_dims) - 1  # number of layers excluding input
        self.epsilon = 1e-8  # For numerical stability in batch normalization

    def initialize_parameters(self):
        np.random.seed(1)
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2. / self.layer_dims[l-1])
            parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
            parameters[f'gamma{l}'] = np.ones((self.layer_dims[l], 1))  # Initialize gamma
            parameters[f'beta{l}'] = np.zeros((self.layer_dims[l], 1))  # Initialize beta
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

    def batch_norm_forward(self, Z, gamma, beta):
        """Perform batch normalization on Z."""
        mean = np.mean(Z, axis=1, keepdims=True)
        variance = np.var(Z, axis=1, keepdims=True)
        Z_norm = (Z - mean) / np.sqrt(variance + self.epsilon)
        Z_tilde = gamma * Z_norm + beta  # Scale and shift
        return Z_tilde, Z_norm, mean, variance

    def forward_propagation(self, X):
        caches = []
        A = X
        
        for l in range(1, self.L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            gamma = self.parameters[f'gamma{l}']
            beta = self.parameters[f'beta{l}']
            
            Z = np.dot(W, A_prev) + b
            Z_tilde, Z_norm, mean, variance = self.batch_norm_forward(Z, gamma, beta)
            
            if l == self.L:  # Output layer
                A = self.softmax(Z_tilde)
            else:
                A = self.relu(Z_tilde)
            
            cache = (A_prev, W, b, Z, Z_norm, mean, variance, gamma, beta)
            caches.append(cache)
        return A, caches

    def batch_norm_backward(self, dZ_norm, Z, mean, variance, gamma):
        """Backward pass for batch normalization."""
        m = Z.shape[1]
        Z_mu = Z - mean
        std_inv = 1. / np.sqrt(variance + self.epsilon)
        
        dZ = (1. / m) * gamma * std_inv * (m * dZ_norm - np.sum(dZ_norm, axis=1, keepdims=True) - Z_mu * std_inv**2 * np.sum(dZ_norm * Z_mu, axis=1, keepdims=True))
        
        dgamma = np.sum(dZ_norm * Z_mu * std_inv, axis=1, keepdims=True)
        dbeta = np.sum(dZ_norm, axis=1, keepdims=True)
        
        return dZ, dgamma, dbeta

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        # Output layer
        dZL = AL - Y
        A_prev, WL, bL, ZL, Z_normL, meanL, varL, gammaL, betaL = caches[self.L - 1]
        grads[f'dW{self.L}'] = (1/m) * np.dot(dZL, A_prev.T)
        grads[f'db{self.L}'] = (1/m) * np.sum(dZL, axis=1, keepdims=True)
        dA_prev, dgammaL, dbetaL = self.batch_norm_backward(dZL, ZL, meanL, varL, gammaL)
        grads[f'dgamma{self.L}'] = dgammaL
        grads[f'dbeta{self.L}'] = dbetaL

        # Hidden layers
        for l in reversed(range(self.L - 1)):
            A_prev, W, b, Z, Z_norm, mean, variance, gamma, beta = caches[l]
            dZ = dA_prev * self.relu_derivative(Z)
            grads[f'dW{l+1}'] = (1/m) * np.dot(dZ, A_prev.T)
            grads[f'db{l+1}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev, dgamma, dbeta = self.batch_norm_backward(dZ, Z, mean, variance, gamma)
            grads[f'dgamma{l+1}'] = dgamma
            grads[f'dbeta{l+1}'] = dbeta

        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
            self.parameters[f'gamma{l}'] -= learning_rate * grads[f'dgamma{l}']
            self.parameters[f'beta{l}'] -= learning_rate * grads[f'dbeta{l}']

    def train(self, X, Y, num_iterations, learning_rate):
        costs = []
        for i in range(num_iterations):
            # Forward propagation
            AL, caches = self.forward_propagation(X)
            
            # Compute cost
            cost = -np.sum(Y * np.log(AL + 1e-8)) / Y.shape[1]
            
            # Backward propagation
            grads = self.backward_propagation(AL, Y, caches)
            
            # Update parameters
            self.update_parameters(grads, learning_rate)
            
            if i % 100 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost}")
        
        return costs

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        return np.argmax(AL, axis=0)

```