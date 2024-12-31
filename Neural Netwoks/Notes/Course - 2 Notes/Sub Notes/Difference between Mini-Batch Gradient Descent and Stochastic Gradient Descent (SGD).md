The key difference between **Mini-Batch Gradient Descent** and **Stochastic Gradient Descent (SGD)** lies in how they update the model's parameters based on the data:

### 1. **Batch Size**:
   - **Stochastic Gradient Descent (SGD)**:
     - **Batch size = 1**. It updates the model's parameters for each individual training example.
     - For each iteration, one random training example is used to compute the gradient and update the parameters.
   - **Mini-Batch Gradient Descent**:
     - **Batch size > 1** but less than the total number of training examples. It divides the dataset into small batches (usually between 32 to 512 samples).
     - For each iteration, a mini-batch of data is used to compute the gradient and update the parameters.

### 2. **Updates per Epoch**:
   - **SGD**:
     - Since SGD processes each training example individually, it performs more frequent updates (one for each training example).
     - If you have `m` training examples, SGD performs `m` updates in one epoch.
   - **Mini-Batch Gradient Descent**:
     - Updates the parameters after processing each mini-batch.
     - If you have `m` training examples and use a mini-batch size of `batch_size`, it performs `m / batch_size` updates per epoch.

### 3. **Noise in the Gradient Estimate**:
   - **SGD**:
     - The gradient estimate is very noisy because it is calculated using only one training example at a time.
     - This can lead to large fluctuations in the cost function and sometimes poor convergence (although it can help in escaping local minima).
   - **Mini-Batch Gradient Descent**:
     - The gradient estimate is less noisy since it's based on multiple examples, giving a more accurate approximation of the true gradient.
     - Less fluctuation in the cost function, leading to smoother and more stable convergence.

### 4. **Speed of Training**:
   - **SGD**:
     - Because it updates the parameters for each training example, SGD can converge faster, but it often fluctuates around the optimal solution rather than converging smoothly.
   - **Mini-Batch Gradient Descent**:
     - It typically converges slower compared to SGD but can achieve a more stable and accurate solution since the gradients are averaged over mini-batches.

### 5. **Memory Usage**:
   - **SGD**:
     - Uses very little memory since it processes one example at a time.
     - This makes SGD more suitable for very large datasets that may not fit into memory.
   - **Mini-Batch Gradient Descent**:
     - Requires more memory compared to SGD because it needs to load a mini-batch into memory at once.
     - However, it's more memory-efficient than full-batch gradient descent.

### 6. **Convergence Behavior**:
   - **SGD**:
     - Typically shows oscillating behavior around the optimal solution because of the high variance in the gradient updates.
     - It might not converge exactly to the minimum but can find a close approximation faster.
   - **Mini-Batch Gradient Descent**:
     - Balances the convergence behavior. The updates are more stable than SGD due to averaging the gradients over multiple examples, leading to smoother convergence.

### 7. **Use Cases**:
   - **SGD**:
     - Suitable for large datasets where fast convergence and lower memory consumption are more critical than precise updates.
     - Can be useful for online learning, where the model is updated continuously with new data.
   - **Mini-Batch Gradient Descent**:
     - A compromise between full-batch gradient descent and SGD.
     - It’s widely used in practice due to its balance between computation efficiency and stable convergence, especially with deep learning frameworks like TensorFlow and PyTorch.

### Summary Table:

| Characteristic          | Stochastic Gradient Descent (SGD) | Mini-Batch Gradient Descent         |
|-------------------------|-----------------------------------|-------------------------------------|
| **Batch Size**           | 1 example                        | Typically between 32 and 512        |
| **Gradient Updates**     | Per training example             | Per mini-batch                     |
| **Noise**                | High                             | Moderate (due to mini-batch averaging) |
| **Memory Usage**         | Low (one example at a time)      | Higher (for mini-batches)           |
| **Convergence Speed**    | Faster but noisy                 | Slower but more stable              |
| **Convergence Behavior** | Oscillates around optimum        | Smooth, more stable convergence     |
| **Use Case**             | Large datasets, online learning  | Deep learning, efficient and stable convergence |

In practice, **mini-batch gradient descent** is the most commonly used method in training neural networks because it provides a good balance between speed, memory usage, and stable convergence. **Stochastic gradient descent** is still useful in specific cases like real-time learning or for very large datasets.