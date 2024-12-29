import numpy as np
from typing import Optional, Tuple, List

class LogisticModel:
    """
    Advanced Logistic Regression implementation with multiple optimization features.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Initial learning rate for gradient descent
    epochs : int, default=1000
        Maximum number of training epochs
    tolerance : float, default=1e-4
        Minimum change in validation loss to qualify as improvement
    patience : int, default=5
        Number of epochs to wait for improvement before early stopping
    batch_size : int or None, default=None
        Size of mini-batches (None for full-batch)
    decay : float, default=0.0
        Learning rate decay factor
    validation_split : float, default=0.2
        Fraction of data to use for validation
    regularization : float, default=0.0
        Regularization strength
    regularization_type : str, default='L2'
        Type of regularization ('L1' or 'L2')
    decay_type : str, default='exponential'
        Type of learning rate decay ('exponential', 'inverse_time', or 'adaptive')
    momentum : float, default=0.0
        Momentum coefficient for gradient descent
    verbose : int, default=1
        Print training progress every verbose epochs
    clip_value : float or None, default=None
        Maximum absolute value for gradient clipping
    random_state : int or None, default=None
        Random seed for reproducibility
    epsilon : float, default=1e-8
        Small constant to prevent division by zero
    init_method: str, default='he'
        Weight initialization method ('he' or 'xavier')
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        tolerance: float = 1e-4,
        patience: int = 20,
        batch_size: Optional[int] = None,
        decay: float = 0.0,
        validation_split: float = 0.2,
        regularization: float = 0.0,
        regularization_type: str = 'L2',
        decay_type: str = 'exponential',
        momentum: float = 0.0,
        verbose: int = 1,
        clip_value: Optional[float] = None,
        random_state: Optional[int] = None,
        epsilon: float = 1e-8,
    ):
        # Validate input parameters
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if validation_split < 0 or validation_split >= 1:
            raise ValueError("Validation split must be between 0 and 1")
        if regularization < 0:
            raise ValueError("Regularization strength must be non-negative")
        if regularization_type not in ['L1', 'L2']:
            raise ValueError("Regularization type must be 'L1' or 'L2'")
        if decay_type not in ['exponential', 'inverse_time', 'adaptive']:
            raise ValueError("Decay type must be 'exponential', 'inverse_time', or 'adaptive'")
            
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.patience = patience
        self.batch_size = batch_size
        self.decay = decay
        self.validation_split = validation_split
        self.regularization = regularization
        self.regularization_type = regularization_type
        self.decay_type = decay_type
        self.momentum = momentum
        self.verbose = verbose
        self.clip_value = clip_value
        self.epsilon = epsilon
        
        if random_state is not None:
            np.random.seed(random_state)
            
        self.weights = None
        self.bias = None
        self.velocity_w = None
        self.velocity_b = None
        self.reset_history()
        
    def reset_history(self) -> None:
        """Reset the training history."""
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute sigmoid function with clipping for stability."""
        z = np.clip(z, -100, 100)  # Adjusted clipping range
        return 1 / (1 + np.exp(-z))
    
    def _initialize_parameters(self, n_features: int) -> None:
        """Initialize weights using Xavier initialization based on input size."""
        # Xavier initialization: sqrt(1/n_in)
        scale = np.sqrt(1. / n_features)
            
        self.weights = np.random.randn(n_features) * scale
        self.bias = 0
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = 0
        
    def _compute_regularization(self) -> float:
        """Compute regularization term."""
        if self.regularization == 0:
            return 0
            
        if self.regularization_type == 'L1':
            return self.regularization * np.sum(np.abs(self.weights))
        return 0.5 * self.regularization * np.sum(self.weights ** 2)  # L2
            
    def _compute_regularization_gradient(self) -> np.ndarray:
        """Compute regularization gradient."""
        if self.regularization == 0:
            return 0
            
        if self.regularization_type == 'L1':
            return self.regularization * np.sign(self.weights)
        return self.regularization * self.weights  # L2
            
    def _update_learning_rate(self, epoch: int, val_loss: float) -> None:
        """Update learning rate based on decay type."""
        if self.decay == 0:
            return
            
        if self.decay_type == 'exponential':
            self.learning_rate = self.initial_learning_rate * np.exp(-self.decay * epoch)
        elif self.decay_type == 'inverse_time':
            self.learning_rate = self.initial_learning_rate / (1 + self.decay * epoch)
        elif self.decay_type == 'adaptive':
            if len(self.history['val_loss']) > 1 and val_loss > self.history['val_loss'][-1]:
                self.learning_rate *= (1 - self.decay)
                
    def _clip_gradients(self, gradient: np.ndarray) -> np.ndarray:
        """Clip gradients to prevent exploding gradients."""
        if self.clip_value is not None:
            if np.isscalar(gradient):
                return max(min(gradient, self.clip_value), -self.clip_value)
            return np.clip(gradient, -self.clip_value, self.clip_value)
        return gradient
        
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Compute binary cross-entropy loss and accuracy."""
        if self.weights is None:
            raise ValueError("Model must be fitted before computing loss")
            
        predictions = self.predict_proba(X)
        epsilon = self.epsilon  # Prevent log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        loss += self._compute_regularization()
        
        accuracy = np.mean((predictions >= 0.5) == y)
        return loss, accuracy
        
    def _get_batches(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate mini-batches for training."""
        if self.batch_size is None:
            return [(X, y)]
            
        indices = np.random.permutation(len(X))
        n_batches = len(X) // self.batch_size
        batches = []
        
        for i in range(n_batches):
            batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
            batches.append((X[batch_indices], y[batch_indices]))
            
        # Handle remaining samples
        if len(X) % self.batch_size != 0:
            batch_indices = indices[n_batches * self.batch_size:]
            batches.append((X[batch_indices], y[batch_indices]))
            
        return batches
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticModel':
        """
        Fit the logistic regression model using mini-batch gradient descent.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (0 or 1)
            
        Returns:
        --------
        self : object
            Returns self.
        """
        X = np.array(X)
        y = np.array(y)
        
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        if len(y.shape) != 1:
            raise ValueError("y must be a 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
            
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)
        
        # Split data into training and validation sets
        val_size = int(self.validation_split * n_samples)
        if val_size > 0:
            X_train, X_val = X[:-val_size], X[-val_size:]
            y_train, y_val = y[:-val_size], y[-val_size:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            batches = self._get_batches(X_train, y_train)
            
            for X_batch, y_batch in batches:
                predictions = self.predict_proba(X_batch)
                
                dw = (1 / len(y_batch)) * X_batch.T @ (predictions - y_batch)
                db = np.mean(predictions - y_batch)
                
                dw += self._compute_regularization_gradient()
                
                dw = self._clip_gradients(dw)
                db = self._clip_gradients(db)
                
                # Update weights with momentum
                self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * dw
                self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * db
                self.weights += self.velocity_w
                self.bias += self.velocity_b
                
            # Evaluate training loss and accuracy
            train_loss, train_accuracy = self._compute_loss(X_train, y_train)
            self.history['train_loss'].append(train_loss) 
            self.history['train_accuracy'].append(train_accuracy)
            
            # Evaluate validation loss and accuracy
            if X_val is not None and y_val is not None:
                val_loss, val_accuracy = self._compute_loss(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                
                # Early stopping check
                if val_loss < best_val_loss - self.tolerance:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
                
            if self.verbose and epoch % self.verbose == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                        f"Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}")
                
            # Learning rate decay
            self._update_learning_rate(epoch, val_loss if X_val is not None else train_loss)
            self.history['learning_rates'].append(self.learning_rate)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for input samples."""
        if self.weights is None:
            raise ValueError("Model must be fitted before predicting")
        return self.sigmoid(X @ self.weights + self.bias)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input samples."""
        return (self.predict_proba(X) >= 0.5).astype(int)
