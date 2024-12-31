
1. **Shuffling the Data**:
   - The data is randomly shuffled to ensure that the mini-batches contain a mix of different examples, which helps the model generalize better.
   - The permutation is applied to both the input data `X` and the labels `Y` so that each training example is still paired with its correct label after shuffling.

2. **Creating Mini-Batches**:
   - After shuffling, the data is divided into mini-batches of size `batch_size`. Each mini-batch will be processed separately during the training loop.
   - `num_batches` represents the total number of mini-batches, calculated by dividing the total number of examples `m` by the batch size.

### Detailed Walkthrough with Example:

Let's assume:
- We have **10 training examples** (`m = 10`), with 2 features per example.
- The **batch size** is set to 3.
- For simplicity, the input data `X` and labels `Y` are represented as follows:

```python
import numpy as np

X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Feature 1
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])  # Feature 2

Y = np.array([[0, 1, 1, 0, 0, 1, 0, 1, 1, 0]])  # Labels
```

The data `X` has 2 features and 10 examples, and `Y` represents the corresponding labels.

#### Step 1: Shuffle the Data
```python
m = X.shape[1]  # m = 10 (number of examples)
permutation = np.random.permutation(m)
print(permutation)  # Example: [3 0 7 9 2 6 5 1 8 4]
```

- `np.random.permutation(m)` generates a random permutation of the indices from `0` to `m-1`. This ensures that the data will be shuffled randomly.
- In this example, the permutation might look like `[3, 0, 7, 9, 2, 6, 5, 1, 8, 4]`.

#### Step 2: Apply the Permutation
```python
X_shuffled = X[:, permutation]
Y_shuffled = Y[:, permutation]
```

- This shuffles both the input data `X` and the labels `Y` according to the generated permutation.
- After shuffling, the data might look like this:

```python
# X_shuffled
[[4, 1, 8, 10, 3, 7, 6, 2, 9, 5],  # Feature 1 (shuffled)
 [7, 10, 3, 1, 8, 4, 5, 9, 2, 6]]  # Feature 2 (shuffled)

# Y_shuffled
[[0, 0, 1, 0, 1, 0, 1, 1, 1, 0]]  # Labels (shuffled)
```

Now, the data is shuffled, but each training example is still paired with its corresponding label.

#### Step 3: Partition into Mini-Batches
```python
batch_size = 3
num_batches = m // batch_size  # num_batches = 10 // 3 = 3 (integer division)
```

- We calculate the number of mini-batches: `num_batches = m // batch_size`. In this case, `m = 10` and `batch_size = 3`, so `num_batches = 3`.

Now, we loop over the number of batches and create mini-batches from the shuffled data.

```python
for t in range(num_batches):
    X_batch = X_shuffled[:, t * batch_size:(t + 1) * batch_size]
    Y_batch = Y_shuffled[:, t * batch_size:(t + 1) * batch_size]
    print(f"Mini-batch {t+1} X:\n{X_batch}")
    print(f"Mini-batch {t+1} Y:\n{Y_batch}")
```

#### Iteration Over Mini-Batches:

**1st Mini-Batch (t = 0):**
- `X_batch = X_shuffled[:, 0:3]`
- `Y_batch = Y_shuffled[:, 0:3]`
- This selects the first 3 examples from the shuffled data.

```python
# X_batch
[[4, 1, 8],  # Feature 1
 [7, 10, 3]]  # Feature 2

# Y_batch
[[0, 0, 1]]  # Labels
```

**2nd Mini-Batch (t = 1):**
- `X_batch = X_shuffled[:, 3:6]`
- `Y_batch = Y_shuffled[:, 3:6]`

```python
# X_batch
[[10, 3, 7],  # Feature 1
 [1, 8, 4]]   # Feature 2

# Y_batch
[[0, 1, 0]]  # Labels
```

**3rd Mini-Batch (t = 2):**
- `X_batch = X_shuffled[:, 6:9]`
- `Y_batch = Y_shuffled[:, 6:9]`

```python
# X_batch
[[6, 2, 9],  # Feature 1
 [5, 9, 2]]  # Feature 2

# Y_batch
[[1, 1, 1]]  # Labels
```

**4th Mini-Batch (t = 3, leftover):**
- Since `m = 10` and `batch_size = 3`, there will be one remaining example (example 10). You may need to handle the last batch if it's smaller than `batch_size`.

### Summary:

- **Shuffling** ensures that each mini-batch contains random samples, preventing the model from learning in a biased order.
- **Partitioning** splits the data into smaller chunks (mini-batches), allowing the model to process and update the parameters more frequently than with full-batch gradient descent.