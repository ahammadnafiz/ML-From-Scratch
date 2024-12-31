# Comprehensive Gradient Descent Breakdown

## What is Gradient Descent?
Gradient Descent is an optimization algorithm used to find the minimum of a function. In machine learning, it's primarily used to minimize the cost function, improving the performance of a model by iteratively adjusting parameters.

## Why Use Gradient Descent?
1. **Optimization**: Helps find the best parameters (weights and biases) for a machine learning model.
2. **Versatility**: Applicable to a wide range of machine learning algorithms.
3. **Efficiency**: Works well with large datasets and high-dimensional problems.
4. **Iterative Improvement**: Gradually improves the model's performance through repeated parameter adjustments.

## Key Components
1. **Cost Function**:  
   -  $J(w,b) = -\frac{1}{m} \sum [y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]$ 
   - Measures how well the model's predictions  $( \hat{y}^{(i)} )$  match the actual data  $( y^{(i)}$ ) .

2. **Model**:  
   - $\hat{y} = \sigma(w^T x + b)$ 
   - Makes predictions based on input features  $( x$ )  and parameters $( w )$ and $( b )$ .

3. **Sigmoid Function**:  
  -  $\sigma(z) = \frac{1}{1 + e^{-z}}$ 
   - Transforms linear combinations into probabilities between 0 and 1.

## How Gradient Descent Works

1. **Initialization**: Start with random values for parameters \( w \) and \( b \).

2. **Forward Pass**: Make predictions using the current parameters.

3. **Cost Calculation**: Compute the cost function J(w,b).

4. **Gradient Computation**: Calculate partial derivatives of \( J \) with respect to \( w \) and \( b \):
$$
  \frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) x^{(i)}
$$
$$
  \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)
$$
[[Derivatives of the cost function J(w,b)]]
5. **Parameter Update**: Adjust \( w \) and \( b \) in the opposite direction of the gradient:
$$
   w := w - \alpha \cdot \frac{\partial J(w,b)}{\partial w} 
$$
$$
    b := b - \alpha \cdot \frac{\partial J(w,b)}{\partial b}
$$
   -  $\alpha$  is the learning rate, controlling the step size.

6. **Iteration**: Repeat steps 2-5 until convergence or for a set number of iterations.

## Visualization

**For a single parameter (1D):**
![[GD_1d.png]]
- The curve represents the cost function  $J(w)$  for different values of $w$.
- Gradient Descent starts at a random point and moves towards the minimum.

**For two parameters (3D):**
![[GD_3d.png]]
- The surface represents  J(w,b)  for different combinations of  w  and b.
- Gradient Descent finds the global minimum (lowest point) on this surface.

## Practical Considerations

1. **Learning Rate**: 
   - If too small, convergence is slow.
   - If too large, the algorithm may overshoot the minimum.

2. **Local Minima**: 
   - For non-convex functions, Gradient Descent can get stuck in local minima.

3. **Variants**:
   - **Stochastic Gradient Descent (SGD)**: Uses one example at a time to update parameters.
   - **Mini-batch Gradient Descent**: Uses small batches of data to update parameters.
   - **Adaptive learning rate methods**: Optimizers like Adam, RMSprop adjust learning rates dynamically.

4. **Convergence**: 
   - Monitor the change in the cost function.
   - Stop when the improvement becomes negligible or after a fixed number of iterations.

[[Step-by-Step Gradient Derivation for Logistic Regression Backpropagation]]