
### Notation and Setup:

-  $X \in \mathbb{R}^{m \times 2}$ : A matrix of input features with shape $(m, 2)$, where  $m$ is the number of training examples and 2 represents the two input features \( $x_1$ \) and \( $x_2$ \).
- $Y \in \mathbb{R}^{m \times 1}$ : A vector of true labels with shape $(m, 1)$.
-  $w \in \mathbb{R}^{2 \times 1}$: A vector of weights for each feature, with shape  $(2, 1)$.
-  $b \in \mathbb{R}$: The bias, which is a scalar.
- $Z \in \mathbb{R}^{m \times 1}$: The linear combination of inputs and weights, with shape  $(m, 1)$ .
-  $A \in \mathbb{R}^{m \times 1}$: The activation (sigmoid output), with shape $(m, 1)$.
-  $alpha$ : The learning rate (a scalar).
-  $J$ : The cost function.

---

### Step-by-Step Breakdown:

#### 1. **Forward Pass**:

We start by computing \( Z \), the linear combination of inputs and weights:


$$
Z = X \cdot w + b
$$


- Shape of \( X \) is $(m \times 2)$ .
- Shape of \( w \) is  $(2 \times 1)$ .
- Shape of \( Z \) will be  $(m \times 1)$ .

Let’s calculate \( Z \) for a single example \( i \):


$$
Z^{(i)} = W_1 \cdot X_1^{(i)} + W_2 \cdot X_2^{(i)} + b
$$


For all examples (vectorized):


$$
Z = 
\begin{pmatrix}
X_1^{(1)} & X_2^{(1)} \\
X_1^{(2)} & X_2^{(2)} \\
\vdots & \vdots \\
X_1^{(m)} & X_2^{(m)}
\end{pmatrix}
\begin{pmatrix}
W_1 \\
W_2
\end{pmatrix}
+ b
$$


#### 2. **Apply the Sigmoid Function**:

Next, we compute the activation \( A \) using the sigmoid function:


$$
A = \sigma(Z) = \frac{1}{1 + e^{-Z}}
$$


- Shape of \( A \) is $(m \times 1)$ since it’s applied element-wise to \( Z \).

For each example \( i \):


$$
A^{(i)} = \frac{1}{1 + e^{-Z^{(i)}}}
$$


---

#### 3. **Compute the Cost Function \( J \)**:

The cost function for logistic regression is:


$$
J = -\frac{1}{m} \sum_{i=1}^{m} \left[ Y^{(i)} \log(A^{(i)}) + (1 - Y^{(i)}) \log(1 - A^{(i)}) \right]
$$


- Shape of \( J \) is a scalar.

For all examples (vectorized):


$$
J = -\frac{1}{m} \left( Y^T \cdot \log(A) + (1 - Y)^T \cdot \log(1 - A) \right)
$$


---

#### 4. **Backward Pass**:

The gradients are calculated to update the weights and bias.

##### 4.1. Compute \( dZ \):


$$
dZ = A - Y
$$


- Shape of \( dZ \) is $(m \times 1)$.

For each example \( i \):


$$
dZ^{(i)} = A^{(i)} - Y^{(i)}
$$


##### 4.2. Compute \( dw \) (Gradient with respect to \( w \)):


$$
dw = \frac{1}{m} X^T \cdot dZ
$$


- Shape of \( $X^T$ \) is \( $(2 \times m)$ \).
- Shape of \( $dZ$ \) is \( $(m \times 1)$ \).
- Shape of \( $dw$ \) is \( $(2 \times 1)$ \).

Let’s calculate the gradient for \( w \):


$$
dw =
\frac{1}{m} \begin{pmatrix}
X_1^{(1)} & X_1^{(2)} & \cdots & X_1^{(m)} \\
X_2^{(1)} & X_2^{(2)} & \cdots & X_2^{(m)}
\end{pmatrix}
\begin{pmatrix}
dZ^{(1)} \\
dZ^{(2)} \\
\vdots \\
dZ^{(m)}
\end{pmatrix}
$$


##### 4.3. Compute \( db \) (Gradient with respect to \( b \)):


$$
db = \frac{1}{m} \sum_{i=1}^{m} dZ^{(i)}
$$


- Shape of \( db \) is a scalar.

For all examples:


$$
db = \frac{1}{m} \sum_{i=1}^{m} (A^{(i)} - Y^{(i)})
$$


---

#### 5. **Gradient Descent Update**:

Now, update the parameters \( w \) and \( b \):


$$
w = w - \alpha \cdot dw
$$


$$
b = b - \alpha \cdot db
$$


- Shape of \( w \) is  $(2 \times 1)$.
- Shape of \( b \) is a scalar.

---

### Summary:

1. **Forward Pass**:
   - Compute \( $Z = X \cdot w + b$ \), where \( $Z \in \mathbb{R}^{m \times 1}$ \).
   - Apply sigmoid: \( $A = \frac{1}{1 + e^{-Z}}$ \), where \( $A \in \mathbb{R}^{m \times 1}$ \).
   
2. **Compute Cost**:
   - Cost function \( $J = -\frac{1}{m} \left[ Y^T \log(A) + (1 - Y)^T \log(1 - A) \right]$ \).

3. **Backward Pass**:
   - \( dZ = A - Y \), where $dZ \in \mathbb{R}^{m \times 1}$.
   -  $dw = \frac{1}{m} X^T \cdot dZ$ , where \( $dw \in \mathbb{R}^{2 \times 1}$ \).
   -  $db = \frac{1}{m} \sum dZ$ , where \( $db \in \mathbb{R}$ \).

4. **Update Parameters**:
   -  $w = w - \alpha \cdot dw$ .
   - $b = b - \alpha \cdot db$.

Here’s the complete **vectorized logistic regression gradient descent** code corresponding to the step-by-step breakdown provided:

```python
import numpy as np

# Assume we have a dataset with m examples (features x1, x2) and labels Y
# X is a matrix with shape (m, 2) where m is the number of examples, and 2 represents the features (x1 and x2).
# Y is a vector with shape (m, 1) representing the labels.

# Initialize parameters
m = X.shape[0]   # number of training examples
n_features = X.shape[1]  # number of features (in this case, 2)

# Weights initialization
w = np.zeros((n_features, 1))  # Initialize weights (w1, w2) to zeros with shape (2, 1)
b = 0                          # Initialize bias to zero (scalar)
alpha = 0.01                   # Learning rate
iterations = 1000              # Number of iterations for gradient descent

for i in range(iterations):
    # Forward pass: Compute Z = X * w + b
    Z = np.dot(X, w) + b      # Z shape: (m, 1)

    # Sigmoid activation: A = 1 / (1 + exp(-Z))
    A = 1 / (1 + np.exp(-Z))  # A shape: (m, 1)

    # Compute the cost function (J)
    J = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # Backward pass
    dZ = A - Y                # dZ shape: (m, 1)
    dw = (1/m) * np.dot(X.T, dZ)  # dw shape: (2, 1)
    db = (1/m) * np.sum(dZ)       # db is a scalar

    # Gradient descent update
    w = w - alpha * dw   # Update weights
    b = b - alpha * db   # Update bias

    # Optionally, print the cost every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}, Cost: {J}")

# Final values of w and b after training
print(f"Trained weights: {w.flatten()}")
print(f"Trained bias: {b}")
```
