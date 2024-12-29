
We will calculate each derivative step by step.

### Given:
- $( a$ ) is the output of the logistic function (or activation function), ( $a = \sigma(z) = \frac{1}{1 + e^{-z}}$ \).
- $( z$ \) is the input to the logistic function, calculated as ( $z = W_1 X_1 + W_2 X_2 + B$ \).
- $( y$ \) is the true label (target).
- $( l$ \) is the loss function, typically for binary cross-entropy:
  
$$
  l = -[y \log(a) + (1 - y) \log(1 - a)]
$$
  

Now, we'll compute the derivatives:

### 1. Derivative of Loss with Respect to \( a \):

$$
\frac{d(l)}{d(a)} = \frac{-y}{a} + \frac{1 - y}{1 - a}
$$


This comes from differentiating the binary cross-entropy loss function with respect to the predicted probability \( a \).

**Step-by-step calculation:**
- The loss function is \( $l = -[y \log(a) + (1 - y) \log(1 - a)]$ \).
- For the term \( $y \log(a)$ \), the derivative with respect to \( a \) is \( $\frac{-y}{a}$ \).
- For the term \( $(1 - y) \log(1 - a)$ \), the derivative with respect to \( a \) is \( $\frac{1 - y}{1 - a}$ \).

Thus,

$$
\frac{d(l)}{d(a)} = \frac{-y}{a} + \frac{1 - y}{1 - a}
$$


### 2. Derivative of Loss with Respect to \( z \):
To find \( $\frac{d(l)}{d(z)}$ \), we apply the chain rule:

$$
\frac{d(l)}{d(z)} = \frac{d(l)}{d(a)} \cdot \frac{d(a)}{d(z)}
$$


We already have \( $\frac{d(l)}{d(a)}$ \) from Step 1. Now, we need \( $\frac{d(a)}{d(z)}$ \).

**Step-by-step calculation for \( $\frac{d(a)}{d(z)}$ \):**
- Recall that \( $a = \sigma(z) = \frac{1}{1 + e^{-z}}$ \).
- The derivative of \( a \) with respect to \( z \) is \( $\sigma'(z) = a(1 - a)$ \), since the derivative of the sigmoid function is \( $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ \).

Thus,

$$
\frac{d(a)}{d(z)} = a(1 - a)
$$


Now, using the chain rule:

$$
\frac{d(l)}{d(z)} = \left(\frac{-y}{a} + \frac{1 - y}{1 - a}\right) \cdot a(1 - a)
$$


However, this expression simplifies directly to:

$$
\frac{d(l)}{d(z)} = a - y
$$


### 3. Derivatives of the Loss with Respect to \( $W_1$ \) and \( $W_2$ \):
For the weights \( $W_1$ \) and \( $W_2$ \), the derivatives are straightforward:


$$
\frac{d(l)}{d(W_1)} = X_1 \cdot \frac{d(l)}{d(z)}
$$


$$
\frac{d(l)}{d(W_2)} = X_2 \cdot \frac{d(l)}{d(z)}
$$


Since we already know \( $\frac{d(l)}{d(z)} = a - y$ \), we substitute this into the equations:


$$
\frac{d(l)}{d(W_1)} = X_1 \cdot (a - y)
$$


$$
\frac{d(l)}{d(W_2)} = X_2 \cdot (a - y)
$$


### 4. Derivative of the Loss with Respect to \( B \):
The bias \( B \) affects the output \( z \) directly, so its derivative is simply:


$$
\frac{d(l)}{d(B)} = \frac{d(l)}{d(z)} = a - y
$$


### Summary of the Derivatives:
1. $\frac{d(l)}{d(a)}$ = ($\frac{-y}{a} + \frac{1 - y}{1 - a}$ \)
2. $\frac{d(l)}{d(z)}$ = ($a - y$ \)
3. $\frac{d(l)}{d(W_1)}$ = $X_1 \cdot (a - y)$
4.  $\frac{d(l)}{d(W_2)}$ = $X_2 \cdot (a - y)$
5. $\frac{d(l)}{d(B)}$ = $a - y$ 
