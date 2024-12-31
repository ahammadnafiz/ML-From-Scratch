### 1. Setting Up the Problem

Consider a binary classification task where we have:

- **Input data**: A set of \( n \) instances, represented by \( $x_1, x_2, \ldots, x_n$ \).
- **True labels**: Corresponding binary labels \( $y_1, y_2, \ldots, y_n$ \), where \( $y_i \in \{0, 1\}$ \).
- **Predicted probabilities**: The model predicts the probability \( $a_i$ \) that each instance belongs to the positive class (i.e., \( y_i = 1 \)). Thus, the probability of the instance belonging to the negative class (i.e., \( $y_i = 0$ \)) is \( $1 - a_i$ \).

### 2. Defining the Likelihood Function

For a single instance, the likelihood of observing the data given the model's predicted probability can be expressed as follows:

- If the true label \( $y_i = 1$ \):
  
$$
  \text{Likelihood}(y_i = 1 | a_i) = a_i
$$
  

- If the true label \( $y_i = 0$ \):
  
$$
  \text{Likelihood}(y_i = 0 | a_i) = 1 - a_i
$$
  

Combining these two cases, we can write the likelihood for a single instance as:

$$
\text{Likelihood}(y_i | a_i) = a_i^{y_i} (1 - a_i)^{(1 - y_i)}
$$

[[Why Use Exponentiation]]
[[Predicted probabilities in the context of binary classification]]
### 3. Extending to Multiple Instances

For \( n \) independent instances, the total likelihood \( L \) of observing all the labels given the predicted probabilities is the product of the likelihoods for each instance:

$$
L = \prod_{i=1}^{n} \text{Likelihood}(y_i | a_i) = \prod_{i=1}^{n} \left(a_i^{y_i} (1 - a_i)^{(1 - y_i)}\right)
$$


### 4. Log-Likelihood

To simplify the optimization process, we take the natural logarithm of the likelihood function, leading to the log-likelihood \( $\log L$ \):

$$
\log L = \sum_{i=1}^{n} \log\left(a_i^{y_i} (1 - a_i)^{(1 - y_i)}\right)
$$


Using properties of logarithms, this can be rewritten as:

$$
\log L = \sum_{i=1}^{n} \left(y_i \log(a_i) + (1 - y_i) \log(1 - a_i)\right)
$$


### 5. Negative Log-Likelihood

In maximum likelihood estimation, we aim to **maximize** the log-likelihood. However, it is often more convenient to **minimize** the negative log-likelihood. Therefore, we define the loss \( l \) as:

$$
l = -\log L = -\sum_{i=1}^{n} \left(y_i \log(a_i) + (1 - y_i) \log(1 - a_i)\right)
$$


For a single instance (to derive the binary cross-entropy loss), we can focus on one example:

$$
l = -[y \log(a) + (1 - y) \log(1 - a)]
$$


### 6. Summary of the Derivation

- The likelihood function is constructed based on the predicted probabilities for both classes.
- The log-likelihood is derived by summing the logarithm of the likelihoods for each instance.
- The loss function is obtained by taking the negative log-likelihood, which effectively penalizes incorrect predictions based on the confidence level of the predictions.

This loss function, \( $l = -[y \log(a) + (1 - y) \log(1 - a)]$ \), captures how well the predicted probabilities \( a \) align with the actual labels \( y \). It is widely used in binary classification problems, particularly in logistic regression.