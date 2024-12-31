In logistic regression, we aim to find parameters \( $\omega$ \) and \( $b$ \) that minimize the difference between the predicted values \( $\hat{y}$ \) and the actual values \( $y$ \). This is done through the **cost function**, which measures the error in the model’s predictions.

---

### 1. **Logistic Regression Model Recap**:
For each training example \( $(x^{(i)}, y^{(i)})$ \), the logistic regression model predicts:

$$
\hat{y}^{(i)} = \sigma(\omega^\top x^{(i)} + b)
$$

Where \( $\sigma(z)$ \) is the **sigmoid function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$


The goal is to minimize the error between \( $\hat{y}^{(i)}$ \) and the actual label \( $y^{(i)}$ \).

---

### 2. **Loss (Error) Function**:
To measure the error for a single training example, we use the **log loss** (also called **binary cross-entropy**):

$$
\mathcal{L}(\hat{y}, y) = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
$$


- If \( $y = 1$ \):
  
$$
  \mathcal{L}(\hat{y}, y) = -\log(\hat{y})
$$
  
  We want \( $\hat{y}$ \) to be close to 1, so \( $\log(\hat{y})$ \) should be large (which happens when \( $\hat{y} \approx 1$ \)).

- If \( $y = 0$ \):
  
$$
  \mathcal{L}(\hat{y}, y) = -\log(1 - \hat{y})
$$
  
  We want ( $\hat{y}$ ) to be close to 0, so ( $\log(1 - \hat{y})$) should be large (which happens when ( $\hat{y} \approx 0$ )).

---

### 3. **Cost Function**:
The **cost function** ( $J(\omega, b)$ ) is the average of the loss function over all ( $m$ ) training examples:

$$
J(\omega, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})
$$

Substitute the log loss for each \( i \)-th example:

$$
J(\omega, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$


This is the cost function used to adjust the parameters \( $\omega$ ) and \( $b$ \) to minimize the error in logistic regression.

---

### 4. **Key Intuition**:
- **When ( $y = 1$ )**: We want ( $\hat{y}$ ) to be as close to 1 as possible, so the term \( $\log(\hat{y})$ \) should be large (which happens when \( $\hat{y}$ \) is close to 1).
- **When ( $y = 0$ )**: We want ( $\hat{y}$ ) to be as close to 0 as possible, so the term \( $\log(1 - \hat{y})$ \) should be large (which happens when \( $\hat{y}$ \) is close to 0).
![[Logistic Regression Cost Function.png]]
Thus, the log loss function penalizes wrong predictions more heavily, ensuring that the model’s predictions improve.

---

### 5. **Summary of the Cost Function**:
- The cost function measures how well the model's predictions match the actual labels across all training examples.
- It is based on the log loss, which provides a smooth gradient that allows for efficient optimization.
- Minimizing the cost function using methods like **gradient descent** will adjust the model parameters ( $\omega$ ) and \( $b$ \) to improve its predictions.

This approach ensures that logistic regression is optimized for binary classification problems, where the output is either 0 or 1.