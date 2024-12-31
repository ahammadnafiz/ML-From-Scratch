
**Key Components:**
1. **Input Layer**: 
   - This layer consists of the input features \(q \).
   - Notation: \( $a^{[0]} = X$ \) where \( $X$ \) represents the input data.

2. **Hidden Layer**:
   - Intermediate layer between the input and output.
   - Neurons in the hidden layer perform weighted summations of the inputs and apply an activation function.
   - Activation of hidden layer neurons is denoted as \( $a^{[1]}$ \), a vector where each element corresponds to a neuron’s activation:
 
$$
     a^{[1]} = \begin{bmatrix}
     a_1^{[1]} \\
     a_2^{[1]} \\
     \vdots \\
     a_4^{[1]} 
     \end{bmatrix}
$$
 
   - Weights \( $W^{[1]}$ \) and biases \( $b^{[1]}$ \) are learned parameters applied to the inputs \( $X$ \) in this layer.

3. **Output Layer**:
   - Produces the predicted output \( $\hat{y} = a^{[2]}$ \).
   - The final neuron activation \( $a^{[2]}$ \) in the output layer is calculated using the weights \( $W^{[2]}$ \) and bias \( $b^{[2]}$ \) applied to the activations from the hidden layer.

**Mathematical Formulation:**
- For the hidden layer, activations are computed as:
  
$$
  z^{[1]} = W^{[1]} X + b^{[1]}
$$
  
  
$$
  a^{[1]} = g(z^{[1]})
$$
  
  where \( g \) is the activation function (e.g., ReLU, sigmoid).
  
- For the output layer:
  
$$
  z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}
$$
  
  
$$
  \hat{y} = a^{[2]} = g(z^{[2]})
$$
  ![[nn_representation.png]]
  Here \( g \) could be a softmax or sigmoid function depending on the task.

**Two-Layer Neural Network:**
- The neural network in this representation is a **2-layer network**, where:
   - The **first layer** is the hidden layer.
   - The **second layer** is the output layer.
