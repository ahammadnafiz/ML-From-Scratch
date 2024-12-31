### General Formula for Forward Propagation

For each layer \( l \) from 1 to \( L \):
1. **Forward propagation for layer \( l \)** (Hidden layer or output layer):

$$
   Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$$
   
   where:
   -  $W^{[l]}$  is the weight matrix for layer \( l \) with shape ($n^{[l]}, n^{[l-1]})$,
   -  $b^{[l]}$  is the bias vector for layer \( l \) with shape ($n^{[l]}, 1$)
   -  $A^{[l-1]}$  is the activation output from the previous layer (for the input layer,  $A^{[0]} = X$ \).

2. **Activation function**:
   - For hidden layers, use **ReLU**:
 
$$
     A^{[l]} = \text{ReLU}(Z^{[l]})
$$
 
   - For the output layer \( l = L \), use **Softmax**:
 
$$
     A^{[L]} = \text{Softmax}(Z^{[L]})
$$
 

The final output of forward propagation is \( $A^{[L]}$ \), which is the predicted output for the network.

### General Formula for Backward Propagation

For each layer \( l \) from \( L \) down to 1 (reverse order):

1. **For the output layer (layer \( L \))**:
   - The error for the output layer is:
 
$$
     dZ^{[L]} = A^{[L]} - Y
$$
 
   where \( Y \) is the true label (shape:  ($n^{[L]}, m$) \).

   - The gradient of weights and biases for layer \( L \):
 
$$
     dW^{[L]} = \frac{1}{m} dZ^{[L]} (A^{[L-1]})^T
$$
 
 
$$
     db^{[L]} = \frac{1}{m} \sum dZ^{[L]} \quad (\text{sum over columns, shape: } (n^{[L]}, 1))
$$
 

   - Backpropagate the gradient into the previous layer:
 
$$
     dA^{[L-1]} = (W^{[L]})^T dZ^{[L]}
$$
 

2. **For hidden layers (layer \( l \) from \( L-1 \) to 1)**:
   - The gradient of \( $Z^{[l]}$ \) (backpropagation through ReLU):
 
$$
     dZ^{[l]} = dA^{[l]} \cdot \text{ReLU'}(Z^{[l]})
$$
 
 where  $\text{ReLU'}(Z^{[l]})$  is the derivative of the ReLU function.
   
   - The gradient of weights and biases for layer \( l \):
 
$$
     dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T
$$
 
 
$$
     db^{[l]} = \frac{1}{m} \sum dZ^{[l]} \quad (\text{sum over columns, shape: } (n^{[l]}, 1))
$$
 

   - Backpropagate the gradient into the previous layer:
 
$$
     dA^{[l-1]} = (W^{[l]})^T dZ^{[l]}
$$
 

### Summary of Formulas

- **Forward propagation**:
  
$$
  Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$$
  
  
$$
  A^{[l]} = \begin{cases}
  \text{ReLU}(Z^{[l]}) & \text{for hidden layers} \\
  \text{Softmax}(Z^{[l]}) & \text{for output layer}
  \end{cases}
$$
  

- **Backward propagation**:
  
$$
  dZ^{[L]} = A^{[L]} - Y
$$
  
  
$$
  dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T, \quad db^{[l]} = \frac{1}{m} \sum dZ^{[l]}
$$
  
  
$$
  dA^{[l-1]} = (W^{[l]})^T dZ^{[l]}
$$
  
$$
  dZ^{[l]} = dA^{[l]} \cdot \text{ReLU'}(Z^{[l]})
$$
  