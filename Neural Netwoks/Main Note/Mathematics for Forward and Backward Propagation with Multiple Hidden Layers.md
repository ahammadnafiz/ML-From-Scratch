Prepared by: Ahammad Nafiz

This template provides a standardized mathematical framework for implementing forward and backward propagation in neural networks with **N hidden layers**. This will be applicable for networks with any number of hidden layers, including both the forward pass (for predictions) and the backward pass (for parameter updates).

---

#### **1. Definitions and Notations**

Let’s define the notations for clarity:

- **L**: Total number of layers, where \( $L = N + 1$ \). (N hidden layers + 1 output layer).
- $n_l​$: Number of neurons in layer $l$, where $l=1,2,…,N+1$ (for N hidden layers and 1 output layer).
- **X**: Input data matrix of shape \($(n_x, m)$\), where:
  - $( n_x$): Number of input features.
  - $( m )$: Number of training examples.
- **Y**: True label matrix of shape \($(n_y, m)$\), where:
  - $( n_y )$: Number of output classes (e.g., 1 for binary classification, C for multiclass).
  - $( m)$: Number of training examples.
- **W[l]**: Weights matrix for layer \(l\), shape depends on layer dimensions.
- **b[l]**: Bias vector for layer \(l\), shape depends on layer dimensions.
- **Z[l]**: Linear transformation at layer \(l\).
- **A[l]**: Activation at layer \(l\).
- **g(·)**: Activation function (ReLU, softmax, etc.).
- **g′(·)**: Derivative of the activation function.
- **α**: Learning rate.

---

### **2. Forward Propagation (General Template)**

For **each hidden layer \( l = 1, 2, ..., N \)** and the output layer \( L = N+1 \), the forward pass involves the following steps:

1. **Linear Transformation**:
   
$$
   Z[l] = W[l]A[l-1] + b[l]
$$
   
   - ( $A[0] = X$ ) (the input layer).
   - ( $Z[l]$ \) is the linear output before applying the activation function.
   -  $W[l] \in \mathbb{R}^{n_l \times n_{l-1}}$, where:
     - ( $n_l$ \): Number of neurons in layer \( l \).
     - ( $n_{l-1}$ \): Number of neurons in the previous layer \( l-1 \).

2. **Activation Function**:
   
$$
   A[l] = g(Z[l])
$$
   
   - For hidden layers, typically use **ReLU**:
 
$$
     g_{\text{ReLU}}(Z^{[l]}) = \max(0, Z^{[l]})
$$

   - For the output layer, typically use **softmax** for classification:
 
$$
     A[L] = g_{softmax}(Z[L])
$$
 
 where \( $g_{softmax}(Z[L])$ \) is a vector of probabilities.

#### **Forward Pass Example for 3 Layers**:
1. **Input Layer → Hidden Layer 1**:
   
$$
   Z[1] = W[1]X + b[1], \quad A[1] = g_{\text{ReLU}}(Z[1])
$$
   
   
2. **Hidden Layer 1 → Hidden Layer 2**:
   
$$
   Z[2] = W[2]A[1] + b[2], \quad A[2] = g_{\text{ReLU}}(Z[2])
$$
   

3. **Hidden Layer 2 → Output Layer**:
   
$$
   Z[3] = W[3]A[2] + b[3], \quad A[3] = g_{softmax}(Z[3])
$$
   

---

### **3. Backward Propagation (General Template)**

Backward propagation starts from the output layer and moves toward the input layer, updating the gradients of weights and biases using the chain rule.

#### **Step-by-Step Backward Propagation:**

For each layer \( l = L, L-1, ..., 1 \):

1. **Output Layer Gradient**:
   The loss gradient with respect to \( Z[L] \):
   
$$
   dZ[L] = A[L] - Y
$$
   
   - This is the difference between predicted probabilities and true labels.

2. **Gradients for Weights and Biases** (Layer \( L \)):
   
$$
   dW[L] = \frac{1}{m} dZ[L] A[L-1]^T
$$
   
   
$$
   db[L] = \frac{1}{m} \sum_{i=1}^{m} dZ[L]
$$
   

3. **Propagate the Error to the Previous Layer** (Hidden Layers):
   For each hidden layer \( l = N, N-1, ..., 1 \):
   
$$
   dZ[l] = W[l+1]^T dZ[l+1] \cdot g_{\text{ReLU}}(Z[l])
$$
   
   - Here  $g_{\text{ReLU}}(Z[l])$ is the derivative of the ReLU activation function:
 
$$
     g_{\text{ReLU}}(Z[l]) = \begin{cases}
     1 & \text{if } Z[l] > 0 \\
     0 & \text{if } Z[l] \leq 0
     \end{cases}
$$
 

4. **Gradients for Weights and Biases** (Hidden Layers):
   For each hidden layer \( l \):
   
$$
   dW[l] = \frac{1}{m} dZ[l] A[l-1]^T
$$
   
   
$$
   db[l] = \frac{1}{m} \sum_{i=1}^{m} dZ[l]
$$
   

#### **Backward Pass Example for 3 Layers**:
1. **Output Layer**:
   
$$
   dZ[3] = A[3] - Y, \quad dW[3] = \frac{1}{m} dZ[3] A[2]^T, \quad db[3] = \frac{1}{m} \sum_{i=1}^{m} dZ[3]
$$
   
   
2. **Hidden Layer 2**:
   
$$
   dZ[2] = W[3]^T dZ[3] \cdot g_{\text{ReLU}}(Z[2]), \quad dW[2] = \frac{1}{m} dZ[2] A[1]^T, \quad db[2] = \frac{1}{m} \sum_{i=1}^{m} dZ[2]
$$
   

3. **Hidden Layer 1**:
   
$$
   dZ[1] = W[2]^T dZ[2] \cdot g_{\text{ReLU}}(Z[1]), \quad dW[1] = \frac{1}{m} dZ[1] X^T, \quad db[1] = \frac{1}{m} \sum_{i=1}^{m} dZ[1]
$$
   

---

### **4. Parameter Updates (General Template)**

For each layer \( l = 1, 2, ..., L \), update the weights and biases using gradient descent:

1. **Weights Update**:
   
$$
   W[l] := W[l] - \alpha dW[l]
$$
   

2. **Biases Update**:
   
$$
   b[l] := b[l] - \alpha db[l]
$$
   

---

### **5. Standard Template for Any Network with N Hidden Layers**

This template can be generalized for a network with **N hidden layers**:

#### **Forward Propagation**:
For each layer \( l = 1, 2, ..., N \):

$$
Z[l] = W[l]A[l-1] + b[l]
$$


$$
A[l] = g_{\text{ReLU}}(Z[l])
$$

For the output layer \( L = N+1 \):

$$
Z[L] = W[L]A[N] + b[L]
$$


$$
A[L] = g_{softmax}(Z[L])
$$


#### **Backward Propagation**:
1. Output layer gradients:
   
$$
   dZ[L] = A[L] - Y
$$
   
   
$$
   dW[L] = \frac{1}{m} dZ[L] A[L-1]^T
$$
   
   
$$
   db[L] = \frac{1}{m} \sum_{i=1}^{m} dZ[L]
$$
   
2. For each hidden layer \( l = N, N-1, ..., 1 \):
   
$$
   dZ[l] = W[l+1]^T dZ[l+1] \cdot g_{\text{ReLU}}(Z[l])
$$
   
   
$$
   dW[l] = \frac{1}{m} dZ[l] A[l-1]^T
$$
   
   
$$
   db[l] = \frac{1}{m} \sum_{i=1}^{m} dZ[l]
$$
   

#### **Parameter Updates**:
For all layers \( l = 1, 2, ..., L \):
1. **Weights Update**:
   
$$
   W[l] := W[l] - \alpha dW[l]
$$
   
2. **Biases Update**:
   
$$
   b[l] := b[l] - \alpha db[l]
$$
   

---

## Matrix Shapes in Multi-Layer Neural Networks: Forward and Backward Propagation

### **1. Forward Propagation Shapes**

For each layer \( l \) in the forward pass, here are the shapes of the key variables:

1. **Input Layer** ( l = 0 ):
   
$$
   X \in \mathbb{R}^{n_x \times m}
$$
   

2. **Hidden Layers** \( $l = 1, 2, \dots, N$ \):
   - **Weights**:
 
$$
     W[l] \in \mathbb{R}^{n_l \times n_{l-1}}
$$
 
   - **Biases**:
 
$$
     b[l] \in \mathbb{R}^{n_l \times 1}
$$
 
   - **Linear Output**:
 
$$
     Z[l] = W[l]A[l-1] + b[l] \quad \Rightarrow \quad Z[l] \in \mathbb{R}^{n_l \times m}
$$
 
   - **Activations**:
 
$$
     A[l] = g(Z[l]) \quad \Rightarrow \quad A[l] \in \mathbb{R}^{n_l \times m}
$$
 
 - **\( A[0] = X \)**: The input layer activation.

3. **Output Layer** \( $l = L = N+1$ \):
   - **Weights**:
 
$$
     W[L] \in \mathbb{R}^{n_y \times n_N}
$$
 
   - **Biases**:
 
$$
     b[L] \in \mathbb{R}^{n_y \times 1}
$$
 
   - **Linear Output**:
 
$$
     Z[L] = W[L]A[N] + b[L] \quad \Rightarrow \quad Z[L] \in \mathbb{R}^{n_y \times m}
$$
 
   - **Activations (Softmax)**:
 
$$
     A[L] = g_{softmax}(Z[L]) \quad \Rightarrow \quad A[L] \in \mathbb{R}^{n_y \times m}
$$
 

---

### **2. Backward Propagation Shapes**

During the backward pass, we compute the gradients of the cost function concerning the weights, biases, and activations. The shapes of these variables are as follows:

1. **Output Layer** \( $l = L = N+1$ \):
   - **Gradient of Loss with respect to \( Z[L] \)**:
 
$$
     dZ[L] = A[L] - Y \quad \Rightarrow \quad dZ[L] \in \mathbb{R}^{n_y \times m}
$$
 
   - **Weight Gradient**:
 
$$
     dW[L] = \frac{1}{m} dZ[L] A[N]^T \quad \Rightarrow \quad dW[L] \in \mathbb{R}^{n_y \times n_N}
$$
 
   - **Bias Gradient**:
 
$$
     db[L] = \frac{1}{m} \sum_{i=1}^{m} dZ[L] \quad \Rightarrow \quad db[L] \in \mathbb{R}^{n_y \times 1}
$$
 

2. **Hidden Layers** (\ $l = N, N-1, \dots, 1$ \):
   - **Gradient of Loss with respect to \( Z[l] \)**:
 
$$
     dZ[l] = W[l+1]^T dZ[l+1] \cdot g_{\text{ReLU}}(Z[l]) \quad \Rightarrow \quad dZ[l] \in \mathbb{R}^{n_l \times m}
$$
 
   - **Weight Gradient**:
 
$$
     dW[l] = \frac{1}{m} dZ[l] A[l-1]^T \quad \Rightarrow \quad dW[l] \in \mathbb{R}^{n_l \times n_{l-1}}
$$
 
   - **Bias Gradient**:
 
$$
     db[l] = \frac{1}{m} \sum_{i=1}^{m} dZ[l] \quad \Rightarrow \quad db[l] \in \mathbb{R}^{n_l \times 1}
$$
 

3. **Input Layer** ($l = 0$ ):
   - The activations from the input layer \( $A[0] = X \in \mathbb{R}^{n_x \times m}$ \).

---

### **3. Parameter Update Shapes**

For each layer \( $l = 1, 2, \dots, L$ \), the weights and biases are updated as follows:

1. **Weights Update**:
   
$$
   W[l] := W[l] - \alpha dW[l] \quad \Rightarrow \quad W[l] \in \mathbb{R}^{n_l \times n_{l-1}}
$$
   

2. **Biases Update**:
   
$$
   b[l] := b[l] - \alpha db[l] \quad \Rightarrow \quad b[l] \in \mathbb{R}^{n_l \times 1}
$$
   

---

### **4. Summary of Shapes**

| Layer | Variable | Shape            |
| ----- | -------- | ---------------- |
| 0     | $X$      | $(n_x, m)$       |
| l     | $W[l]$   | $(n_l, n_{l-1})$ |
| l     | $b[l]$   | $(n_l, 1)$       |
| l     | $Z[l]$   | $(n_l, m)$       |
| l     | $A[l]$   | $(n_l, m)$       |
| L     | $dZ[L]$  | $(n_y, m)$       |
| l     | $dZ[l]$  | $(n_l, m)$       |
| l     | $dW[l]$  | $(n_l, n_{l-1})$ |
| l     | $db[l]$  | $(n_l, 1)$       |
| L     | $dW[L]$  | $(n_y, n_N)$     |
| L     | $db[L]$  |  $(n_y, 1)$      |
