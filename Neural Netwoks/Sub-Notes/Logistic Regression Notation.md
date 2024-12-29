
In logistic regression, we aim to model the probability that a binary outcome \( y \) (e.g., 0 or 1) occurs, given the input feature vector \( x \). The model outputs a probability between 0 and 1.

#### 1. **Objective**:
We want to estimate the probability:

$$
\hat{y} = P(y = 1 | x)
$$

Where:
- $( x \in \mathbb{R}^{n_x} )$ is the input feature vector with \( $n_x$ \) features.
- $( \hat{y} )$ is the predicted probability that \( $y = 1$ \), bounded between 0 and 1: \( $0 \leq \hat{y} \leq 1$ \).

#### 2. **Parameters**:
- The model parameters consist of:
  - **Weights** \( $\omega \in \mathbb{R}^{n_x}$ \): These represent the importance of each feature.
  - **Bias** ( $b \in \mathbb{R}$ ): This is a scalar that allows shifting the decision boundary.

#### 3. **Hypothesis Function**:
The model's prediction is given by the logistic (sigmoid) function:

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \omega^\top x + b
$$

Where:
- $( z = \omega^\top x + b$ \) is a linear combination of the input features and the weights.
- $( \sigma(z)$ ) is the sigmoid function, which maps any real number \( z \) to the range \( (0, 1) \).

#### 4. **Sigmoid Behavior**:
- For large positive values of \( z \), \( $\sigma(z) \to 1$ \).
- For large negative values of ( z ), ( $\sigma(z) \to 0$ ).

This allows the model to output probabilities for binary classification.

---

### Logistic Regression in Matrix Form

1. **Augmented Feature Vector**:
- We can rewrite the logistic regression model by augmenting the feature vector \( x \) to include the bias term. 
- Let \( $x_0 = 1$ \), so:
  
$$
  x \in \mathbb{R}^{n_x + 1}, \quad \hat{y} = \sigma(\Theta^\top x)
$$
  
  Here, \( $\Theta = [b, \omega_1, \omega_2, \dots, \omega_{n_x}]$ \) is the parameter vector, with the bias \( b \) as its first element.

2. **Parameter Vector \( \Theta \)**:
- $( \Theta \in \mathbb{R}^{n_x + 1}$ ) consists of:
  - $( \Theta_0 = b)$, the bias term.
  - $( \Theta_1, \Theta_2, \dots, \Theta_{n_x} = \omega )$, the weights for the features.

---

### Sigmoid Function

The sigmoid function \( \sigma(z) \) is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$


- When \( z \) is a large positive number:
  
$$
  \sigma(z) \approx 1
$$
  
- When \( z \) is a large negative number:
  
$$
  \sigma(z) \approx 0
$$
  

This ensures the output ( $\hat{y}$ ) is always between 0 and 1, making it suitable for binary classification.

---

### Practical Use in Logistic Regression

Given an input feature vector \( x \), logistic regression calculates a linear combination \( $z = \omega^\top x + b$ \) and applies the sigmoid function to produce a probability:
\[
$$
\hat{y} = \sigma(\omega^\top x + b)
$$
\]

This allows us to classify the input as ( y = 1 ) if ( $\hat{y} \geq 0.5$ ), or ( $y = 0$ ) if ( $\hat{y} < 0.5$ ).

---
[[Logistic Regression Cost Function]]