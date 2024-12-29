import numpy as np
from math import ceil
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError
from typing import Optional, Literal, Tuple, Union, List

class LinearModel(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        tolerance: float = 1e-4,
        patience: int = 30,
        batch_size: Optional[int] = 64,
        decay: float = 0.995,
        validation_split: float = 0.2,
        regularization: float = 1e-4,
        regularization_type: Literal['L1', 'L2'] = 'L2',
        decay_type: Literal['exponential', 'inverse_time', 'adaptive'] = 'eadaptive',
        momentum: float = 0.9,
        verbose: int = 0,
        clip_value: Optional[float] = 1.0,
        random_state: Optional[int] = None,
        epsilon: float = 1e-8
    ):
        """
        Initialize the Linear Regression model with improved parameters.
        
        Args:
            learning_rate: Initial learning rate for gradient descent
            epochs: Maximum number of training epochs
            tolerance: Minimum change in validation loss to qualify as improvement
            patience: Number of epochs to wait for improvement before early stopping
            batch_size: Size of mini-batches (None for full-batch)
            decay: Learning rate decay factor
            validation_split: Fraction of data to use for validation
            regularization: Regularization strength
            regularization_type: Type of regularization ('L1' or 'L2')
            decay_type: Type of learning rate decay ('exponential', 'inverse_time', or 'adaptive')
            momentum: Momentum coefficient for gradient descent
            verbose: Print training progress every verbose epochs
            clip_value: Maximum absolute value for gradient clipping
            random_state: Random seed for reproducibility
            epsilon: Small constant to prevent division by zero
        """
        self._validate_inputs(
            learning_rate, epochs, tolerance, patience, 
            validation_split, regularization, decay, momentum
        )
        
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
        self.random_state = random_state
        self.epsilon = epsilon
        
        # Initialize model parameters
        self.w = None
        self.b = None
        self.velocity_w = None
        self.velocity_b = None
        self.history = []
        self.val_history = []
        self._is_fitted = False

    def _validate_inputs(
        self,
        learning_rate: float,
        epochs: int,
        tolerance: float,
        patience: int,
        validation_split: float,
        regularization: float,
        decay: float,
        momentum: float
    ) -> None:
        """Validate input parameters with improved error messages."""
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if not isinstance(patience, int) or patience < 0:
            raise ValueError("patience must be a non-negative integer")
        if not isinstance(validation_split, float) or not 0 <= validation_split < 1:
            raise ValueError("validation_split must be a float between 0 and 1")
        if not isinstance(regularization, (int, float)) or regularization < 0:
            raise ValueError("regularization must be non-negative")
        if not isinstance(decay, float) or not 0 < decay <= 1:
            raise ValueError("decay must be a float between 0 and 1")
        if not isinstance(momentum, float) or not 0 <= momentum < 1:
            raise ValueError("momentum must be a float between 0 and 1")

    def _compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        include_regularization: bool = True
    ) -> Tuple[float, np.ndarray, float]:
        """
        Compute the loss and gradients with improved numerical stability.
        
        Args:
            X: Input features
            y: Target values
            include_regularization: Whether to include regularization term
        """
        y_pred = np.dot(X, self.w) + self.b
        m = X.shape[0]
        
        # Compute MSE loss and gradients with numerical stability
        residuals = y - y_pred
        loss = np.mean(np.square(residuals))
        dw = (-2 / m) * np.dot(X.T, residuals)
        db = (-2 / m) * np.sum(residuals)
        
        if include_regularization and self.regularization > 0:
            if self.regularization_type == 'L2':
                reg_loss = 0.5 * self.regularization * np.sum(np.square(self.w))
                reg_dw = self.regularization * self.w
            else:  # L1
                reg_loss = self.regularization * np.sum(np.abs(self.w))
                reg_dw = self.regularization * np.sign(self.w)
            
            loss += reg_loss
            dw += reg_dw
        
        return loss + self.epsilon, dw, db

    def _learning_schedule(self, t: int, current_loss: float = None, prev_loss: float = None) -> float:
        """
        Compute learning rate based on schedule type and current state.
        
        Args:
            t: Current iteration number
            current_loss: Current loss value (for adaptive decay)
            prev_loss: Previous loss value (for adaptive decay)
        """
        if self.learning_rate < 1e-10:
            self.learning_rate = 1e-10
        
        if self.decay_type == 'exponential':
            return self.learning_rate * (self.decay ** t)
        elif self.decay_type == 'inverse_time':
            t0, t1 = 200, 1000  # hyperparameters for inverse time decay
            return self.learning_rate * t0 / (t + t1)
        else:  # adaptive  
            if current_loss is not None and prev_loss is not None:
                if current_loss > prev_loss:
                    self.learning_rate *= self.decay
                elif current_loss < prev_loss * (1 - self.tolerance):
                    self.learning_rate = min(self.learning_rate / self.decay, self.initial_learning_rate)
            return self.learning_rate

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

    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialize model parameters using Xavier/Glorot initialization.
        
        Args:
            n_features: Number of input features
        """
        limit = np.sqrt(6 / (n_features + 1))
        self.w = np.random.uniform(-limit, limit, n_features)
        self.b = 0
        self.velocity_w = np.zeros_like(self.w)
        self.velocity_b = 0

    def fit(self, X: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> 'LinearModel':
        """
        Fit the linear regression model using improved gradient descent.
        
        Args:
            X: Training features
            y: Training targets
        """
        # Convert inputs to numpy arrays
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # Input validation
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Ensure enough samples for validation
        min_train_samples = 2
        if n_samples < min_train_samples / (1 - self.validation_split):
            raise ValueError(
                f"Not enough samples for validation split. Need at least "
                f"{ceil(min_train_samples / (1 - self.validation_split))} samples."
            )
        
        # Split data into training and validation sets
        split_idx = int((1 - self.validation_split) * n_samples)
        indices = np.random.permutation(n_samples)
        X_train, X_val = X[indices[:split_idx]], X[indices[split_idx:]]
        y_train, y_val = y[indices[:split_idx]], y[indices[split_idx:]]
        
        # Initialize parameters
        self._initialize_parameters(n_features)
        self.history = []
        self.val_history = []
        
        best_loss = float('inf')
        patience_counter = 0
        prev_loss = None
        
        for epoch in range(self.epochs):
            # Get batch indices
            batches = self._get_batches(X_train, y_train)
            
            epoch_loss = 0
            for X_batch, y_batch in batches:
                # Compute gradients
                batch_loss, dw, db = self._compute_loss(X_batch, y_batch)
                epoch_loss += batch_loss
                
                # Apply gradient clipping
                if self.clip_value is not None:
                    dw = np.clip(dw, -self.clip_value, self.clip_value)
                    db = np.clip(db, -self.clip_value, self.clip_value)
                
                # Update learning rate
                lr = self._learning_schedule(epoch, batch_loss, prev_loss)
                prev_loss = batch_loss
                
                # Update parameters with momentum
                self.velocity_w = self.momentum * self.velocity_w - lr * dw
                self.velocity_b = self.momentum * self.velocity_b - lr * db
                self.w += self.velocity_w
                self.b += self.velocity_b
            
            # Compute average epoch loss and validation loss
            avg_epoch_loss = epoch_loss / len(batches)
            self.history.append(avg_epoch_loss)
            
            # Early stopping with improved relative improvement check
            if X_val is not None and y_val is not None:
                val_loss, _, _ = self._compute_loss(X_val, y_val, include_regularization=False)
                self.val_history.append(val_loss)
                # Early stopping check
                if val_loss < best_loss - self.tolerance:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            if self.verbose and epoch % self.verbose == 0:
                print(f"Epoch {epoch}, Training Loss: {avg_epoch_loss:.4f}, "
                      f"Validation Loss: {val_loss:.4f}, Learning Rate: {lr:.6f}")
        
        self._is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Make predictions for the input features.
        
        Args:
            X: Input features
        """
        if not self._is_fitted:
            raise NotFittedError(
                "This LinearRegressionScratch instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        
        X = np.array(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        if X.shape[1] != len(self.w):
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with {len(self.w)} features"
            )
            
        return np.dot(X, self.w) + self.b

    def learning_curve(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot the learning curves with improved visualization.
        
        Args:
            figsize: Figure size as (width, height)
        """
        if not self._is_fitted:
            raise NotFittedError(
                "This LinearRegressionScratch instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )
            
        plt.figure(figsize=figsize)
        epochs = range(1, len(self.history) + 1)
        
        plt.plot(epochs, self.history, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_history, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.show()