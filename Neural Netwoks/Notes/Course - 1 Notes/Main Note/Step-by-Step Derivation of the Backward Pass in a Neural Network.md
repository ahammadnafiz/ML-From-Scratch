Prepared By: Ahammad Nafiz
[ Blog](https://t.co/bG4uL0ADq4)
[Linkedin](https://www.linkedin.com/in/ahammad-nafiz/)

![[backpropmath.png]]
### **Part 1: Output Layer Backward Pass**

We’ll derive each of these:
1. \( $dz^{[L]}$ \) (the gradient of the loss with respect to the pre-activation \( $z^{[L]}$ \))
2. \( $dW^{[L]}$ \) (the gradient of the loss with respect to the weights \( $W^{[L]}$ \))
3. \( $db^{[L]}$ \) (the gradient of the loss with respect to the biases \( $b^{[L]}$ \))
4. \( $dA^{[L-1]}$ \) (the gradient of the loss with respect to the activations from the previous layer \( $A^{[L-1]}$ \))

#### **Step 1: Deriving \( $dz^{[L]}$ \)**

The output layer uses the softmax activation function, and we compute the loss using cross-entropy. The loss function is given as:

$$
\mathcal{L} = - \frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{n_L} Y_k^{(i)} \log(A_k^{[L](i)})
$$

Where:
- \( Y \) is the true label matrix (one-hot encoded),
- \( $A^{[L]}$ \) is the output from the softmax layer,
- \( m \) is the number of training examples,
- \( $n_L$ \) is the number of output units (number of classes).

Now, the softmax activation is applied to \( $z^{[L]}$ \), the pre-activation of the output layer:

$$
A^{[L]} = \text{Softmax}(z^{[L]})
$$

Where the softmax function for class \( k \) is:

$$
A_k^{[L]} = \frac{e^{z_k^{[L]}}}{\sum_{j=1}^{n_L} e^{z_j^{[L]}}}
$$


Now, to compute the gradient \( $dz^{[L]}$ \), we first recall that:


$$
dz^{[L]} = \frac{\partial \mathcal{L}}{\partial z^{[L]}}
$$


Using the property of softmax combined with cross-entropy loss, the gradient simplifies to:

$$
dz^{[L]} = A^{[L]} - Y
$$

This is because:
- \( $A^{[L]}$ \) is the predicted probability,
- \( Y \) is the one-hot encoded true labels.

This is the difference between the predicted output and the true label, which will be the error term at the output layer.

#### **Step 2: Deriving \( $dW^{[L]}$ \)**

Next, we derive the gradient of the loss with respect to the weights \( $W^{[L]}$ \). We need to calculate:


$$
dW^{[L]} = \frac{\partial \mathcal{L}}{\partial W^{[L]}}
$$

From the forward pass, we know that:

$$
z^{[L]} = W^{[L]} A^{[L-1]} + b^{[L]}
$$

So, by the chain rule:


$$
dW^{[L]} = \frac{\partial \mathcal{L}}{\partial z^{[L]}} \cdot \frac{\partial z^{[L]}}{\partial W^{[L]}}
$$


We already calculated \( $\frac{\partial \mathcal{L}}{\partial z^{[L]}} = dz^{[L]}$ \).

Now, \( $\frac{\partial z^{[L]}}{\partial W^{[L]}} = A^{[L-1]}$ \), the activations from the previous layer. Thus:

$$
dW^{[L]} = \frac{1}{m} dz^{[L]} A^{[L-1]T}
$$


This expression shows that the gradient with respect to the weights is the average of the product of the error term \( $dz^{[L]}$ \) and the activations from the previous layer \( $A^{[L-1]}$ \), transposed.

#### **Step 3: Deriving \( $db^{[L]}$ \)**

The gradient with respect to the bias \( $b^{[L]}$ \) is simpler to compute. We have:


$$
db^{[L]} = \frac{\partial \mathcal{L}}{\partial b^{[L]}}
$$


Again, using the chain rule:

$$
db^{[L]} = \frac{\partial \mathcal{L}}{\partial z^{[L]}} \cdot \frac{\partial z^{[L]}}{\partial b^{[L]}}
$$


Since \( $\frac{\partial z^{[L]}}{\partial b^{[L]}} = 1$ \), the derivative simplifies to:


$$
db^{[L]} = \frac{1}{m} \sum_{i=1}^{m} dz^{[L](i)}
$$


This is the average of the error terms \( $dz^{[L]}$ \) across all examples.

#### **Step 4: Deriving \( $dA^{[L-1]}$ \)**

Now we calculate the gradient with respect to the activations from the previous layer \( $A^{[L-1]}$ \):


$$
dA^{[L-1]} = \frac{\partial \mathcal{L}}{\partial A^{[L-1]}}
$$


Again, using the chain rule:

$$
dA^{[L-1]} = \frac{\partial \mathcal{L}}{\partial z^{[L]}} \cdot \frac{\partial z^{[L]}}{\partial A^{[L-1]}}
$$


From the forward pass, we know:

$$
z^{[L]} = W^{[L]} A^{[L-1]} + b^{[L]}
$$

So:

$$
\frac{\partial z^{[L]}}{\partial A^{[L-1]}} = W^{[L]}
$$


Thus:

$$
dA^{[L-1]} = W^{[L]T} dz^{[L]}
$$


This shows that the gradient with respect to \( $A^{[L-1]}$ \) is the product of the transposed weight matrix \( $W^{[L]T}$ \) and the error term \( $dz^{[L]}$ \).

---

### **Part 2: Hidden Layer Backward Pass (General Case)**

Now we repeat this process for a hidden layer \( $l$ \) where \( $1 \leq l \leq L-1$ \). For the hidden layers, we use the ReLU activation function.

We need to derive:
1. \( $dz^{[l]}$ \)
2. \( $dW^{[l]}$ \)
3. \( $db^{[l]}$ \)
4. \( $dA^{[l-1]}$ \)

#### **Step 1: Deriving \( $dz^{[l]}$ \)**

At the hidden layers, the activation function is ReLU. We start by computing the gradient of the loss with respect to \( $z^{[l]}$ \).

From the forward pass:

$$
A^{[l]} = \text{ReLU}(z^{[l]})
$$


The derivative of the ReLU function is:

$$
\frac{d}{dz^{[l]}} \text{ReLU}(z^{[l]}) =
\begin{cases} 
1 & \text{if } z^{[l]} > 0 \\
0 & \text{if } z^{[l]} \leq 0 
\end{cases}
$$


So, the derivative of the loss with respect to \( $z^{[l]}$ \) is:

$$
dz^{[l]} = dA^{[l]} \odot \text{ReLU}'(z^{[l]})
$$

Where:
- \( $\odot$ \) is the element-wise (Hadamard) product,
- \( $dA^{[l]}$ \) is the gradient from the next layer,
- \( $\text{ReLU}'(z^{[l]})$ \) is the derivative of the ReLU function.

**Why Element-wise Product?**

The ReLU function operates element-wise on the matrix \( $z^{[l]}$ \) (meaning, it acts on each element independently). Specifically:

$$
\text{ReLU}(z) = \max(0, z)
$$


The derivative of ReLU, \( $\text{ReLU}'(z^{[l]})$ \), is also computed element-wise:

$$
\text{ReLU}'(z^{[l]}) = 
\begin{cases}
1 & \text{if } z^{[l]} > 0 \\
0 & \text{if } z^{[l]} \leq 0
\end{cases}
$$


When we propagate the gradient backward through this activation function, we only want to "pass through" the gradient \( $dA^{[l]}$ \) where the ReLU was active (i.e., where \( $z^{[l]} > 0$ \)).

This is why we do an **element-wise product**. It ensures that for each element of \( $z^{[l]}$ \):
- If \( $z^{[l]} > 0$ \), the gradient flows through (i.e., it's multiplied by 1),
- If \( $z^{[l]} \leq 0$ \), the gradient is stopped (i.e., it's multiplied by 0).

### **How Does it Work?**

When we compute \( $dz^{[l]}$ \), we're combining the gradient from the next layer \( $dA^{[l]}$ \) with the local gradient of the ReLU activation function:

$$
dz^{[l]} = dA^{[l]} \odot \text{ReLU}'(z^{[l]})
$$


- \( $dA^{[l]}$ \) contains the gradients coming from the next layer (how much the loss changes with respect to the activation output),
- \( $\text{ReLU}'(z^{[l]})$ \) indicates where ReLU is active or not.

Thus, the element-wise product ensures that:
- For activations where \( $z^{[l]} > 0$ \) (where the neuron is active), the gradient is preserved and passed back,
- For activations where \( $z^{[l]} \leq 0$ \), the gradient is zeroed out, preventing it from flowing back.
[[Why Element Wise Multiplication]]
#### **Step 2: Deriving \( $dW^{[l]}$ \)**

Now, we compute the gradient with respect to the weights \( $W^{[l]}$ \):


$$
dW^{[l]} = \frac{\partial \mathcal{L}}{\partial W^{[l]}}
$$


Using the chain rule:

$$
dW^{[l]} = \frac{\partial \mathcal{L}}{\partial z^{[l]}} \cdot \frac{\partial z^{[l]}}{\partial W^{[l]}}
$$


From the forward pass, we know:

$$
z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$$

Thus:

$$
dW^{[l]} = \frac{1}{m} dz^{[l]} A^{[l-1]T}
$$


#### **Step 3: Deriving \( $db^{[l]}$ \)**

The gradient with respect to the bias \( $b^{[l]}$ \) is:


$$
db^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dz^{[l](i)}
$$


####

 **Step 4: Deriving \( $dA^{[l-1]}$ \)**

Finally, we compute the gradient with respect to the activations from the previous layer \( $A^{[l-1]}$ \):


$$
dA^{[l-1]} = W^{[l]T} dz^{[l]}
$$


This completes the backward pass for the hidden layer.

---

### **Summary of the Backward Pass Derivations**

1. **Output Layer**:
   - \( $dz^{[L]} = A^{[L]} - Y$ \)
   - \( $dW^{[L]} = \frac{1}{m} dz^{[L]} A^{[L-1]T}$ \)
   - \( $db^{[L]} = \frac{1}{m} \sum_{i=1}^{m} dz^{[L](i)}$ \)
   - \( $dA^{[L-1]} = W^{[L]T} dz^{[L]}$ \)

2. **Hidden Layers**:
   - \( $dz^{[l]} = dA^{[l]} \odot \text{ReLU}'(z^{[l]})$ \)
   - \( $dW^{[l]} = \frac{1}{m} dz^{[l]} A^{[l-1]T}$ \)
   - \( $db^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dz^{[l](i)}$ \)
   - \( $dA^{[l-1]} = W^{[l]T} dz^{[l]}$ \)