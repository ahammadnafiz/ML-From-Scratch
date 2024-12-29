
#### 1. **Inputs and Weights**
- A **perceptron** is a simple neural network model that processes multiple inputs.
- Each input ( $x_1, x_2, \dots, x_n$ ) has an associated **weight** ( $w_1, w_2, \dots, w_n$ ).
- The perceptron computes the **weighted sum** of inputs:

  $$ z = W^T \cdot X = w_1x_1 + w_2x_2 + \dots + w_nx_n $$

  where \( $W^T$ \) is the transposed vector of weights and \( X \) is the input vector.

#### 2. **Bias Term**
- A **bias** \( $b$ \) is often added to the weighted sum:

  $$ z = W^T \cdot X + b $$

- The bias helps shift the decision boundary, allowing the perceptron to make better predictions.

#### 3. **Activation Function**
- The perceptron uses an **activation function** to decide whether to fire (output 1) or not (output 0).
- The **step function** is the most basic activation function:

  $$ \text{output} = \begin{cases} 
  1 & \text{if } z > \text{threshold} \\
  0 & \text{if } z \leq \text{threshold}
  \end{cases} $$

- The **threshold** can be adjusted based on the problem, but often it's set to 0.

#### 4. **Perceptron as a Binary Classifier**
- The perceptron acts as a **binary classifier**.
- It learns a **decision boundary** (a line in 2D, or a plane in higher dimensions) to separate data into two classes.
- If the weighted sum \( z \) exceeds the threshold, the perceptron **"fires"** and outputs 1. Otherwise, it outputs 0.

#### 5. **Key Takeaways**
- **Weighted sum**: \( $W^T \cdot X$ \) represents how much influence each input has based on its weight.
- **Bias**: Helps adjust the output independently of the inputs.
- **Threshold**: Determines the point at which the perceptron "fires" or stays off.
- **Step function**: Acts as the decision-maker, determining the output of the perceptron.