# Machine Learning: Comprehensive Notes

## 1. Introduction
Machine Learning (ML) is the field of study that enables computers to learn without explicit programming.

## 2. Types of Learning

### 2.1 Supervised Learning
- Training set includes desired solutions (labels)
- Examples: Classification, Regression

### 2.2 Unsupervised Learning
- Training data is unlabeled
- Subtypes:
  - **Dimensionality Reduction**: Simplify data while preserving information
  - **Novelty Detection**: Identify new instances differing from training set
  - **Association Rule Learning**: Discover attribute relationships in large datasets

### 2.3 Semi-Supervised Learning
- Algorithms work with partially labeled data

### 2.4 Self-Supervised Learning
- Generate fully labeled dataset from unlabeled data
- Apply supervised learning algorithms after labeling

### 2.5 Reinforcement Learning
- Agent observes environment, performs actions, receives rewards/penalties

## 3. Learning Approaches

### 3.1 Batch Learning
- System trained using all available data
- Incapable of incremental learning

### 3.2 Online Learning (Incremental Learning)
- System trained incrementally with sequential data instances
- Key parameter: learning rate (adaptation speed to changing data)

## 4. Generalization Methods

### 4.1 Instance-Based Learning
- System memorizes examples
- Generalizes using similarity measures

### 4.2 Model-Based Learning
- Example: Simple Linear Model
  ```
  life_satisfaction = θ0 + θ1 × GDP_per_capita
  ```

## 5. Challenges in Machine Learning

### 5.1 Data Quality Issues
- Sampling noise: Non-representative data due to small sample size
- Sampling bias: Flawed sampling method even in large samples

### 5.2 Feature Engineering
- Critical for project success
- Process of creating good training features

### 5.3 Overfitting
- Model performs well on training data but doesn't generalize
- Occurs when model is too complex relative to data
- Solution: Regularization (constraining model to reduce overfitting)
  - Controlled by hyperparameters

### 5.4 Underfitting
- Model too simple to learn data's underlying structure

## 6. Testing and Validation

### 6.1 Data Splitting
- Training set: For model learning
- Test set: For final model evaluation

### 6.2 Error Metrics
- Generalization error (out-of-sample error): Error rate on new cases

### 6.3 Validation Techniques
- Holdout validation: Reserve part of training set for model selection
- Cross-validation: Use multiple small validation sets
- Repeated cross-validation: For more robust model selection

## 7. Key Takeaways
1. ML enables machines to improve through data-driven learning
2. Various ML system types exist (supervised/unsupervised, batch/online, instance-based/model-based)
3. ML projects involve:
   - Data gathering
   - Training set creation
   - Algorithm application (model-based or instance-based)
4. System performance depends on:
   - Training set size and quality
   - Feature relevance
   - Model complexity balance (avoid underfitting/overfitting)
