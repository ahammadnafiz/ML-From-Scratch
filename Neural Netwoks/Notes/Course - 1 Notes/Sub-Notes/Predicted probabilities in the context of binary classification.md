
### 1. What Are Predicted Probabilities?

In binary classification problems, the goal is to classify an instance into one of two classes, typically labeled as:

- **Positive class**: Often represented by \( y = 1 \)
- **Negative class**: Often represented by \( y = 0 \)

A model, such as logistic regression or a neural network, outputs a probability \( $a_i$ \) for each instance \( i \). This probability represents the model's belief that the instance belongs to the positive class. 

### 2. Understanding the Prediction

- **Output \( $a_i$ \)**: The predicted probability that instance \( i \) belongs to the positive class (i.e., \( $y_i = 1$ \)).
  - \( $a_i$ \) is a continuous value in the range \($[0, 1]$\).
  - If \( $a_i$ \) is close to 1, the model is very confident that the instance is positive.
  - If \( $a_i$ \) is close to 0, the model is very confident that the instance is negative.

- **Probability of the Negative Class**: The probability that the instance belongs to the negative class (i.e., \( $y_i = 0$ \)) can be derived from the predicted probability of the positive class:
  
$$
  P(y_i = 0 | x_i) = 1 - a_i
$$
  
  - This means if the model predicts a high probability for the positive class (close to 1), the probability for the negative class will be low (close to 0).
  - Conversely, if the model predicts a low probability for the positive class (close to 0), the probability for the negative class will be high (close to 1).

### 3. Probabilistic Interpretation

The use of probabilities allows the model to express uncertainty:

- **Thresholding**: Typically, a threshold (often 0.5) is used to make a final classification decision based on the predicted probability:
  - If \( $a_i \geq 0.5$ \), classify the instance as positive (\( $y_i = 1$ \)).
  - If \( $a_i < 0.5$ \), classify the instance as negative (\( $y_i = 0$ \)).
  
This probabilistic output is beneficial for various reasons:

- **Confidence Levels**: It provides insight into how confident the model is in its predictions. A predicted probability of 0.9 indicates high confidence, whereas a probability of 0.51 indicates uncertainty.
  
- **Handling Uncertainty**: In cases where the model is uncertain (e.g., predictions near the threshold), you can take additional measures, such as gathering more data, adjusting model parameters, or choosing different thresholds based on the problem context.

### 4. Example

Let’s consider an example to illustrate:

Suppose you have a model that predicts whether an email is spam (positive class, \( y = 1 \)) or not spam (negative class, \( y = 0 \)). 

- For a specific email, the model predicts:
  - \( $a_1 = 0.8$ \): The model predicts an 80% probability that the email is spam.
    - This means \( $P(y_1 = 0 | x_1) = 1 - 0.8 = 0.2$ \): There's a 20% probability it is not spam.
  - If another email has \( $a_2 = 0.3$ \): The model predicts a 30% probability that it is spam.
    - Thus, \( $P(y_2 = 0 | x_2) = 1 - 0.3 = 0.7$ \): There's a 70% probability it is not spam.

### Summary

In summary, predicted probabilities allow a binary classification model to express its predictions in a nuanced way, reflecting its confidence in classifying instances as positive or negative. This approach enables better decision-making and provides flexibility in handling uncertainty in the predictions.