The Mini-Batch Gradient Descent algorithm is a variation of the gradient descent method, where, instead of using the full dataset for each iteration (batch gradient descent) or one sample at a time (stochastic gradient descent), we use a small subset of the data for each iteration. This helps in speeding up the training process while still approximating the performance of the full batch method.

### Mini-Batch Gradient Descent Algorithm:
Here's a pseudo-code for one epoch of training using mini-batches.

```python
# Given:
# X -> input data, Y -> true labels
# initialize_parameters() -> initializes weights and biases
# forward_prop(X, Y) -> computes the forward pass
# compute_cost(AL, Y) -> computes the cost function
# backward_prop(AL, caches) -> computes the gradients using backpropagation
# update_parameters(grads) -> updates weights and biases

# Parameters:
# No_of_batches: Total number of mini-batches in the dataset
# batch_size: Number of samples in each mini-batch
# t: Mini-batch index for the current iteration

initialize_parameters()

for t = 1:No_of_batches:   # For each mini-batch in the epoch
    # Step 1: Forward propagation
    AL, caches = forward_prop(X[t], Y[t])

    # Step 2: Compute cost
    cost = compute_cost(AL, Y[t])

    # Step 3: Backward propagation (Gradient calculation)
    grads = backward_prop(AL, caches)

    # Step 4: Update parameters (Weights and biases)
    update_parameters(grads)

    # Optional: Monitor training progress (e.g., print cost every few iterations)
    if t % 100 == 0:
        print(f"Cost after batch {t}: {cost}")
```

### Detailed Explanation:

1. **Why Mini-Batch?**  
   Mini-batch gradient descent offers a balance between efficiency and performance:
   - **Full-batch gradient descent** is computationally expensive for large datasets and can get stuck in local minima.
   - **Stochastic gradient descent (SGD)** is faster but more noisy, leading to larger fluctuations in the cost.
   - **Mini-batch gradient descent** balances the noise and stability, providing faster convergence with less computational load than full-batch.

2. **Key Steps in the Algorithm**:
   - **Initialize Parameters**: We randomly initialize the weights and biases.
   - **Loop over each mini-batch (1 epoch)**:
     1. **Forward Propagation**: Compute predictions (`AL`) for the current mini-batch of inputs `X[t]` and their corresponding labels `Y[t]`.
     2. **Cost Calculation**: Calculate the cost (loss) using the predicted output `AL` and the true labels `Y[t]`. The cost function depends on the task (e.g., cross-entropy for classification).
     3. **Backward Propagation**: Compute gradients of the cost function with respect to the parameters by backpropagating the error through the network.
     4. **Update Parameters**: Use an optimization method (e.g., gradient descent, Adam) to update the weights and biases using the computed gradients.

3. **Formulas**:
   - **Cost Function**: For a classification task, the cross-entropy cost function can be computed as:
 
$$
     J = -\frac{1}{m} \sum_{i=1}^{m} \left( Y_i \log(AL_i) + (1 - Y_i) \log(1 - AL_i) \right)
$$

where \(m\) is the number of samples in the mini-batch.
   - **Parameter Update**: In gradient descent, we update the parameters as follows:
     
$$
     W = W - \alpha \cdot \nabla W
$$
     
     where \(W\) represents the weights, \($\alpha$\) is the learning rate, and \($\nabla W$\) is the gradient with respect to the weights.

4. **When to Use Mini-Batch Gradient Descent**:
   - **When dataset is large**: Mini-batch gradient descent is suitable for large datasets where full-batch gradient descent would be too slow.
   - **To speed up convergence**: Mini-batches allow for faster training compared to using the whole dataset at once.
   - **When hardware resources are limited**: Mini-batches fit well in memory compared to the entire dataset, making them practical for real-world training scenarios.

### Summary of Key Points:
- **Mini-batch** size is critical: Too small can make training noisy, too large may be computationally expensive.
- **Learning rate**: Adjust carefully when using mini-batch gradient descent, as improper settings can lead to divergence.
- **Batch shuffling**: Each epoch, shuffle the dataset to create different mini-batches, improving model generalization.

Here's a Python implementation of the Mini-Batch Gradient Descent algorithm:

```python
import numpy as np

class NN:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters()
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

    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']

    def create_mini_batches(self, X, Y, batch_size):
        m = X.shape[1]
        mini_batches = []
        
        # Shuffle data
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        # Create mini-batches
        num_batches = m // batch_size
        for i in range(num_batches):
            X_batch = X_shuffled[:, i * batch_size: (i + 1) * batch_size]
            Y_batch = Y_shuffled[:, i * batch_size: (i + 1) * batch_size]
            mini_batches.append((X_batch, Y_batch))
        
        # Handle remaining samples if they don't fit perfectly into batches
        if m % batch_size != 0:
            X_batch = X_shuffled[:, num_batches * batch_size:]
            Y_batch = Y_shuffled[:, num_batches * batch_size:]
            mini_batches.append((X_batch, Y_batch))

        return mini_batches

    def train(self, X, Y, num_iterations, learning_rate, batch_size):
        costs = []
        for i in range(num_iterations):
            mini_batches = self.create_mini_batches(X, Y, batch_size)
            for X_batch, Y_batch in mini_batches:
                # Forward propagation
                AL, caches = self.forward_propagation(X_batch)
                
                # Compute cost
                cost = -np.sum(Y_batch * np.log(AL + 1e-8)) / Y_batch.shape[1]
                
                # Backward propagation
                grads = self.backward_propagation(AL, Y_batch, caches)
                
                # Update parameters
                self.update_parameters(grads, learning_rate)
            
            # Print cost every 100 iterations
            if i % 100 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost}")
        
        return costs

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        return np.argmax(AL, axis=0)

```

[[Logic Explanation]]
### Key Points:
1. **Data Shuffling**: The dataset is shuffled at the beginning of each epoch to ensure that the model generalizes well.
2. **Mini-Batch Partitioning**: The data is divided into mini-batches of the specified size (`batch_size`). Each mini-batch is used for one forward and backward pass.
3. **Cost Monitoring**: The cost is printed every 100 epochs to monitor the progress of the training.
4. **Learning Rate**: The learning rate controls the step size in gradient descent. You can tune this based on your specific problem.