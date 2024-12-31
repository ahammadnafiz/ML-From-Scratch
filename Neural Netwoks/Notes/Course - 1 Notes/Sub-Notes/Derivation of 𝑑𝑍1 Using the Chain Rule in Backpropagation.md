### Recap of the Network Structure and Variables:
-  $A2 = \sigma(Z2)$ , where ( $\sigma$ ) is the sigmoid function applied to the pre-activation $( Z2$ \).
-  $Z2 = W2 \cdot A1 + b2$ , where \( A1 \) is the activation from the first hidden layer.
-  $A1 = \sigma(Z1)$ , where \( $Z1 = W1 \cdot A0 + b1$ \) (and \( $A0 = X$ \)).

Now, we want to calculate \( dZ1 \) using the chain rule, which will involve propagating the derivative of the cost function with respect to the output back to \( Z1 \).

### Step 1: Start with the derivative of the cost function
The derivative of the cost function \( J \) with respect to \( Z2 \) is given as:

$$
dZ2 = \frac{\partial J}{\partial Z2} = A2 - Y
$$

This comes from applying the chain rule on the cost function \( J \) with respect to the predictions \( A2 \).

### Step 2: Apply the chain rule to \( A1 \) with respect to \( Z1 \)
Next, we need to propagate this error back to \( Z1 \) by applying the chain rule.

We know:

$$
Z2 = W2 \cdot A1 + b2
$$


Thus, by the chain rule:

$$
\frac{\partial J}{\partial A1} = \frac{\partial J}{\partial Z2} \cdot \frac{\partial Z2}{\partial A1}
$$


Since:

$$
\frac{\partial Z2}{\partial A1} = W2^T
$$


We get:

$$
\frac{\partial J}{\partial A1} = W2^T \cdot dZ2
$$


### Step 3: Chain rule for \( Z1 \)
Now that we have \( $\frac{\partial J}{\partial A1}$ \), we apply the chain rule again to move from \( A1 \) to \( Z1 \).

Recall that \( $A1 = \sigma(Z1)$ \), so the derivative of \( A1 \) with respect to \( Z1 \) is:

$$
\frac{\partial A1}{\partial Z1} = \sigma'(Z1)
$$

For the sigmoid activation function, the derivative \( $\sigma'(Z1)$ \) is:

$$
\sigma'(Z1) = A1 \cdot (1 - A1)
$$


Thus, applying the chain rule:

$$
dZ1 = \frac{\partial J}{\partial Z1} = \frac{\partial J}{\partial A1} \cdot \frac{\partial A1}{\partial Z1}
$$


### Step 4: Putting it all together
Now, substitute the terms we derived:


$$
dZ1 = (W2^T \cdot dZ2) \cdot \sigma'(Z1)
$$

where \( $\sigma'(Z1) = A1 \cdot (1 - A1)$ \), which is the derivative of the sigmoid function applied element-wise.

So the final expression for \( dZ1 \) is:

$$
dZ1 = (W2^T \cdot dZ2) \cdot A1 \cdot (1 - A1)
$$


### Summary:
1. $dZ2 = A2 - Y$ 
2. $dZ1 = (W2^T \cdot dZ2) \cdot A1 \cdot (1 - A1)$