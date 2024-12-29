When training a neural network, choosing the right activation function is crucial for efficient learning. Let's explore **ReLU** and **Sigmoid** in terms of their impact on training, focusing on the **Vanishing Gradient Problem**, **gradient flow**, and their mathematical properties.

---

### 1. **ReLU Activation Function**
- **Definition**: The Rectified Linear Unit (ReLU) is defined as:

$$
   f(x) = \max(0, x)
$$

- **Derivative** (for backpropagation):

$$
  f'(x) =
  \begin{cases} 
  1, & x > 0 \\
  0, & x \leq 0 
  \end{cases}
$$

- **Advantages**:
  1. **No Vanishing Gradient**: For positive inputs, the derivative is always 1, which avoids the vanishing gradient issue.
  2. **Sparse Activation**: Neurons only activate when \( x > 0 \), leading to sparse activations, which makes the network more efficient.
  3. **Faster Convergence**: The constant gradient of 1 for \( x > 0 \) allows for better gradient flow, accelerating training.
  
- **Disadvantage**: **Dying ReLU**: Neurons with negative input are always zero, and once they become inactive, they never recover. This is known as the "dying ReLU" problem, but it's often mitigated by variations like **Leaky ReLU**.

### 2. **Sigmoid Activation Function**
- **Definition**: The sigmoid activation function maps inputs to a range between 0 and 1.

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

- **Derivative** (for backpropagation):

$$
  f'(x) = f(x) (1 - f(x))
$$

- **Advantages**:
  1. **Saturated Output**: The output is always bounded between 0 and 1, which can be useful for binary classification.
  
- **Disadvantages**:
  1. **Vanishing Gradient**: When \( x \) is very large or very small, the gradient of the sigmoid function becomes close to 0, causing the **vanishing gradient problem**. This makes it difficult for the network to learn, especially in deeper layers.
  2. **Non-sparse Activation**: All neurons are always active (i.e., have non-zero output), making the network less efficient compared to ReLU.

---

### **Vanishing Gradient Problem: Sigmoid vs. ReLU**

In deep networks, gradients are backpropagated through multiple layers. The gradient of the sigmoid function diminishes as it propagates backward, especially when neurons are in their saturated zones (large positive or negative inputs). This makes it harder to train deep networks.

**ReLU** solves this problem for the following reasons:
- For positive values of \( x \), ReLU’s gradient is 1, so gradients can flow through the network without shrinking.
- The non-zero gradient for positive values helps in updating the weights effectively, even in deep networks.

#### **Mathematical Insight: Vanishing Gradient in Sigmoid**
For large positive or negative inputs, \( f'(x) \) for sigmoid tends to 0:

$$
When ( x \to +\infty ), ( f(x) \to 1 ) and ( f'(x) \to 0 )
$$
$$
 When ( x \to -\infty ), ( f(x) \to 0 ) and ( f'(x) \to 0 )
$$

Thus, gradients become very small, causing slow learning or no learning in deep layers.

---

### **Code Examples: ReLU vs. Sigmoid**

#### **ReLU Example**

![[relu.png]]
![[relu_dx.png]]
#### **Sigmoid Example**

![[sigmoid.png]]

![[sigmoid_dx.png]]
---

### **Gradient Flow in Deep Networks**
The sigmoid function tends to cause smaller gradients, especially in deeper layers, resulting in slower learning. The ReLU function keeps gradients alive for positive values of the input, thus allowing deeper networks to learn faster and more effectively.

#### **Example of Vanishing Gradient with Sigmoid**

```python
# Example showing vanishing gradients with sigmoid in deeper layers

# Sigmoid derivative becomes very small for large or small inputs
gradients = []
x = np.linspace(-10, 10, 100)

# Calculate gradients at each layer (in deep layers)
for layer in range(1, 6):
    grad = sigmoid_derivative(x)
    gradients.append(grad)
    x = grad  # Forward pass input to next layer is the gradient of the current layer

# Plot gradients through layers
plt.plot(np.linspace(-10, 10, 100), gradients[-1], label=f"Layer {layer}")
plt.title("Vanishing Gradient with Sigmoid in Deep Networks")
plt.legend()
plt.grid(True)
plt.show()
```

---

### **Summary: Why ReLU is Preferred**

1. **Mitigates Vanishing Gradient Problem**: ReLU ensures that the gradient doesn’t vanish as it does with sigmoid, allowing efficient training in deep networks.
2. **Sparse Activation**: Only a fraction of neurons activate at any given time, making it computationally efficient.
3. **Simple Computation**: ReLU is simpler to compute than sigmoid, making the forward and backward passes faster.

**ReLU** has become the standard activation function in most modern neural networks, especially in deep learning architectures like convolutional neural networks (CNNs) and deep feed-forward networks.