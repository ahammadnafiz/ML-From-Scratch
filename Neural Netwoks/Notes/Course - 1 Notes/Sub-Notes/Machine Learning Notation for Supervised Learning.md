
In supervised learning, the data consists of input features and corresponding labels, structured as follows:

#### 1. **Training Example Structure**:
Each training example is represented by an input-output pair \($(x^{(i)}, y^{(i)})$\), where:

- $x^{(i)} \in \mathbb{R}^{n_x}$: The input feature vector for the ($i^{th}$) example, where \($n_x$\) is the number of features.
- \($y^{(i)} \in \{0,1\}$\): The output label corresponding to the input \($x^{(i)}$\). In a binary classification problem, the label is either 0 or 1.
- The full training set contains \($m$\) examples: 
  
$$
  \{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \ldots, (x^{(m)}, y^{(m)})\}
$$
  
  
#### 2. **Dataset Representation**:
The inputs and outputs for the entire training dataset are often represented as matrices.

- **Input Matrix \(X\)**:
  
$$
  X = \begin{bmatrix}
  x^{(1)} & x^{(2)} & \dots & x^{(m)}
  \end{bmatrix} \in \mathbb{R}^{n_x \times m}
$$
  
  - Each column \($x^{(i)}$\) corresponds to the feature vector of the \($i^{th}$\) training example.
  - The matrix \(X\) has dimensions \($n_x \times m$\), where:
    - \($(n_x$\) is the number of features.
    - $(m)$\) is the number of training examples.

- **Output Matrix \(Y\)**:
  
$$
  Y = \begin{bmatrix}
  y^{(1)} & y^{(2)} & \dots & y^{(m)}
  \end{bmatrix} \in \mathbb{R}^{1 \times m}
$$
  
  - Each entry $(y^{(i)}$\) corresponds to the label of the \($i^{th}$\) training example.
  - The matrix \(Y\) has dimensions \($1 \times m$\), where:
    - $(m$) is the number of training examples.

#### 3. **Training and Test Sets**:
- $(M_{\text{train}} = m$): The total number of training examples.
- $(M_{\text{test}})$: The total number of test examples.

This structure allows machine learning algorithms to efficiently process and model the relationship between inputs \(X\) and outputs \(Y\) in order to make predictions on new data.
