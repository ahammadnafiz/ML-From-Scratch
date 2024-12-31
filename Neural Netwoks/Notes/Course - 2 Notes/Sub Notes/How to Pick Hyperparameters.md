
#### When to Use
When tuning hyperparameters that span several orders of magnitude, such as learning rates or regularization coefficients, using a linear scale for searching values may not be ideal. Instead, applying a logarithmic scale is more effective. This approach is particularly relevant for hyperparameters like learning rates, weight decay, or parameters such as momentum in neural networks. 

#### Why Use a Logarithmic Scale
Hyperparameters such as learning rates or regularization terms often have optimal values that vary exponentially. Searching over these parameters in a linear scale could miss potential optimal values due to non-uniform exploration across the range. A logarithmic scale ensures a more balanced and thorough exploration by uniformly sampling values on an exponential scale. This improves the likelihood of discovering the best-performing hyperparameters.

#### How to Use a Logarithmic Scale
Given a range for a hyperparameter from \( a \) to \( b \), where \( a \) and \( b \) are bounds on the hyperparameter value, you can convert these bounds into a logarithmic scale to ensure a more even sampling:

1. **Convert the range into logarithmic scale**:
   
$$
   a_{\text{log}} = \log(a) \quad \text{and} \quad b_{\text{log}} = \log(b)
$$
   

2. **Generate a random sample between the logarithmic values**:
   
$$
   r = (a_{\text{log}} - b_{\text{log}}) \times \text{rand}(0,1) + b_{\text{log}}
$$
   
   
   Here, `rand(0,1)` generates a random number between 0 and 1.
   
3. **Convert back to the original scale**:
   
$$
   \text{result} = 10^r
$$
   
   This gives a sampled value within the desired range on the original scale.

#### Example: Learning Rate
Let’s assume the optimal learning rate is expected to fall between \( a = 0.0001 \) and \( b = 1 \). Using the logarithmic scale, you can search this range as follows:

-  $a_{\text{log}} = \log(0.0001) = -4$ 
-  $b_{\text{log}} = \log(1) = 0$ 
- Sample  $r = (-4 - 0) \times \text{rand}(0,1) + 0$ , which gives  $r \in [-4, 0]$ 
- Convert back: $\text{learning rate} = 10^r$ 

This method ensures the learning rate is uniformly sampled between \( 0.0001 \) and \( 1 \), providing better coverage over the range.

#### Example: Momentum Beta
Momentum’s hyperparameter \( \beta \) typically ranges from 0.9 to 0.999, but directly sampling within this range could result in uneven exploration. Instead, search for \( 1 - \beta \) in the range 0.001 to 0.1, which corresponds to \( \beta \in [0.9, 0.999] \):

- $a = 0.001 ,  b = 0.1$ 
- $a_{\text{log}} = \log(0.001) = -3 ,  b_{\text{log}} = \log(0.1) = -1$ 
- Sample  $r = (-3 - (-1)) \times \text{rand}(0,1) + (-1)$ 
- Calculate \( $1 - \beta = 10^r$ \), hence \( $\beta = 1 - 10^r$ \)

This technique samples \( \beta \) values uniformly in log space between 0.9 and 0.999, making the search more effective.

#### Key Points
- **Logarithmic search**: Use log scale for hyperparameters spanning several orders of magnitude.
- **Applications**: Suitable for learning rates, momentum, regularization terms, and similar parameters.
- **Balanced exploration**: Ensures uniform sampling in the hyperparameter’s exponential space.