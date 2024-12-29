To find the derivatives of the cost function $J(w, b)$ with respect to  $w$ and  $b$ , we need to perform the following steps. Let’s start with the cost function you provided:

$$
 J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right], 
$$

where \( $\hat{y}^{(i)} = \sigma(z^{(i)})$ \) is the predicted probability, \( $z^{(i)} = w^T x^{(i)} + b$ \) is the linear combination of the inputs, and \( $\sigma(z) = \frac{1}{1 + e^{-z}}$ \) is the sigmoid function.

**1. Derivative with respect to \( w \):**

First, let's compute the derivative of \( J \) with respect to \( w \). We need to apply the chain rule:

- Compute the partial derivative of \( J \) with respect to \( $\hat{y}^{(i)}$ \):
  
$$
  \frac{\partial J}{\partial \hat{y}^{(i)}} = -\frac{1}{m} \left[ \frac{y^{(i)}}{\hat{y}^{(i)}} - \frac{1 - y^{(i)}}{1 - \hat{y}^{(i)}} \right]
$$
  

- Compute the partial derivative of \( $\hat{y}^{(i)}$ \) with respect to \( $z^{(i)}$ \):
  
$$
  \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} = \hat{y}^{(i)} \left( 1 - \hat{y}^{(i)} \right)
$$
  

- Compute the partial derivative of \( $z^{(i)}$ \) with respect to \( w \):
  
$$
  \frac{\partial z^{(i)}}{\partial w} = x^{(i)}
$$
  

Combining these using the chain rule:

$$
\frac{\partial J}{\partial w} = \sum_{i=1}^{m} \frac{\partial J}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial w}
$$


$$
\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) x^{(i)}
$$


**2. Derivative with respect to \( b \):**

Now, compute the derivative of \( J \) with respect to \( b \). Follow these steps:

- Compute the partial derivative of \( J \) with respect to \( $\hat{y}^{(i)}$ \) (same as above):
  
$$
  \frac{\partial J}{\partial \hat{y}^{(i)}} = -\frac{1}{m} \left[ \frac{y^{(i)}}{\hat{y}^{(i)}} - \frac{1 - y^{(i)}}{1 - \hat{y}^{(i)}} \right]
$$
  

- Compute the partial derivative of \( $\hat{y}^{(i)}$ \) with respect to \( $z^{(i)}$ \) (same as above):
  
$$
  \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} = \hat{y}^{(i)} \left( 1 - \hat{y}^{(i)} \right)
$$
  

- Compute the partial derivative of \( $z^{(i)}$ \) with respect to \( $b$ \):
  
$$
  \frac{\partial z^{(i)}}{\partial b} = 1
$$
  

Combining these:

$$
\frac{\partial J}{\partial b} = \sum_{i=1}^{m} \frac{\partial J}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial b}
$$


$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)
$$


To summarize:

- **Derivative with respect to \( w \)**:
  
$$
  \frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) x^{(i)}
$$
  

- **Derivative with respect to \( b \)**:
  
$$
  \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)
$$