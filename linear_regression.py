import numpy as np
import warnings

class LinearRegression:
    """
    Linear Regression with optional L1 (Lasso) and L2 (Ridge) regularization.
    Implemented purely using NumPy.
    """
    def __init__(self, penalty=None, alpha=1.0, lr=0.01, n_iters=1000, solver='gd', loss='mse', 
                 missing_strategy='mean', drop_missing=False):
        """
        Parameters:
        -----------
        penalty : str or None
            Regularization penalty. Can be None (OLS), 'l1' (Lasso), or 'l2' (Ridge).
        alpha : float
            Regularization strength.
        lr : float
            Learning rate for Gradient Descent solver.
        n_iters : int
            Number of iterations for Gradient Descent solver.
            'gd' (Gradient Descent) or 'closed' (Closed-form solution).
            Note: 'closed' is only available for None and 'l2' penalty.
        loss : str
            Loss function to optimize. Can be 'mse' (Mean Squared Error) or 'mae' (Mean Absolute Error).
            Note: 'closed' solver only supports 'mse'.
        missing_strategy : str
            Strategy for handling missing values. Options: 'mean', 'median', 'forward_fill', 'backward_fill'.
            Default: 'mean'.
        drop_missing : bool
            If True, drop rows with any missing values. If False, use missing_strategy. Default: False.
        """
        if penalty not in [None, 'l1', 'l2']:
            raise ValueError("Penalty must be None, 'l1', or 'l2'")
        if solver not in ['gd', 'closed']:
            raise ValueError("Solver must be 'gd' or 'closed'")
        if solver == 'closed' and penalty == 'l1':
            raise ValueError("Closed-form solution is not available for L1 penalty (Lasso). Use solver='gd'.")
        if loss not in ['mse', 'mae']:
            raise ValueError("Loss must be 'mse' or 'mae'")
        if solver == 'closed' and loss != 'mse':
            raise ValueError("Closed-form solver only supports 'mse' loss.")
        if missing_strategy not in ['mean', 'median', 'forward_fill', 'backward_fill']:
            raise ValueError("Missing strategy must be 'mean', 'median', 'forward_fill', or 'backward_fill'")

        self.penalty = penalty
        self.alpha = alpha
        self.lr = lr
        self.n_iters = n_iters
        self.solver = solver
        self.loss = loss
        self.missing_strategy = missing_strategy
        self.drop_missing = drop_missing
        
        self.weights = None
        self.bias = None
        self.feature_means = None  # Store means for imputation during prediction
        self.feature_medians = None  # Store medians for imputation during prediction

    def _handle_missing_values(self, X, y=None, fit=False):
        """
        Handle missing values in features and target.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix, may contain NaN values.
        y : array-like of shape (n_samples,), optional
            Target values, may contain NaN values.
        fit : bool
            If True, compute and store imputation statistics (means/medians).
            If False, use pre-computed statistics from training data.
        
        Returns:
        --------
        X_clean : array of shape (n_samples, n_features)
            Features with missing values handled.
        y_clean : array of shape (n_samples,), or None
            Target with missing values handled, or None if y is None.
        """
        X = np.array(X, dtype=float)
        
        # Count missing values
        n_missing = np.isnan(X).sum()
        if n_missing > 0:
            warnings.warn(f"Found {n_missing} missing values in X. Using {self.missing_strategy} imputation.")
        
        if self.drop_missing:
            # Drop rows with any missing values in X
            valid_idx = ~np.isnan(X).any(axis=1)
            X_clean = X[valid_idx]
            if y is not None:
                y = np.array(y, dtype=float)
                # Also drop corresponding y values with NaN
                valid_idx_y = ~np.isnan(y)
                valid_idx = valid_idx & valid_idx_y
                X_clean = X[valid_idx]
                y_clean = y[valid_idx]
            else:
                y_clean = None
        else:
            # Use imputation strategy
            if fit:
                # Compute and store statistics
                self.feature_means = np.nanmean(X, axis=0)
                self.feature_medians = np.nanmedian(X, axis=0)
            
            X_clean = X.copy()
            
            if self.missing_strategy == 'mean':
                for col in range(X_clean.shape[1]):
                    mask = np.isnan(X_clean[:, col])
                    if mask.any():
                        X_clean[mask, col] = self.feature_means[col]
            
            elif self.missing_strategy == 'median':
                for col in range(X_clean.shape[1]):
                    mask = np.isnan(X_clean[:, col])
                    if mask.any():
                        X_clean[mask, col] = self.feature_medians[col]
            
            elif self.missing_strategy == 'forward_fill':
                for col in range(X_clean.shape[1]):
                    mask = np.isnan(X_clean[:, col])
                    if mask.any():
                        # Forward fill: use previous valid value
                        valid_idx = np.where(~mask)[0]
                        if len(valid_idx) > 0:
                            for i in range(X_clean.shape[0]):
                                if mask[i]:
                                    prev_valid = valid_idx[valid_idx < i]
                                    if len(prev_valid) > 0:
                                        X_clean[i, col] = X_clean[prev_valid[-1], col]
                                    else:
                                        # No previous value, use mean
                                        X_clean[i, col] = self.feature_means[col]
            
            elif self.missing_strategy == 'backward_fill':
                for col in range(X_clean.shape[1]):
                    mask = np.isnan(X_clean[:, col])
                    if mask.any():
                        # Backward fill: use next valid value
                        valid_idx = np.where(~mask)[0]
                        if len(valid_idx) > 0:
                            for i in range(X_clean.shape[0]):
                                if mask[i]:
                                    next_valid = valid_idx[valid_idx > i]
                                    if len(next_valid) > 0:
                                        X_clean[i, col] = X_clean[next_valid[0], col]
                                    else:
                                        # No next value, use mean
                                        X_clean[i, col] = self.feature_means[col]
            
            if y is not None:
                y = np.array(y, dtype=float)
                y_clean = y.copy()
                # Drop rows where y has NaN
                valid_idx = ~np.isnan(y_clean)
                X_clean = X_clean[valid_idx]
                y_clean = y_clean[valid_idx]
            else:
                y_clean = None
        
        return X_clean, y_clean

    def fit(self, X, y):
        """
        Fit the linear regression model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        # Handle missing values
        X, y = self._handle_missing_values(X, y, fit=True)
        
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape

        if self.solver == 'closed':
            # Add bias term (column of 1s) to X for closed-form computation
            X_b = np.c_[np.ones((n_samples, 1)), X]

            if self.penalty is None:
                # OLS: w = (X^T * X)^-1 * X^T * y
                theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            elif self.penalty == 'l2':
                # Ridge: w = (X^T * X + alpha * I)^-1 * X^T * y
                I = np.eye(X_b.shape[1])
                I[0, 0] = 0  # Do not regularize the bias term
                theta = np.linalg.pinv(X_b.T.dot(X_b) + self.alpha * I).dot(X_b.T).dot(y)

            self.bias = theta[0]
            self.weights = theta[1:]

        elif self.solver == 'gd':
            # Initialize parameters
            self.weights = np.zeros(n_features)
            self.bias = 0.0

            for _ in range(self.n_iters):
                y_predicted = np.dot(X, self.weights) + self.bias

                # Compute gradients based on selected loss function
                if self.loss == 'mse':
                    dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
                    db = (1 / n_samples) * np.sum(y_predicted - y)
                elif self.loss == 'mae':
                    dw = (1 / n_samples) * np.dot(X.T, np.sign(y_predicted - y))
                    db = (1 / n_samples) * np.sum(np.sign(y_predicted - y))

                # Add regularization term to gradient
                if self.penalty == 'l2':
                    dw += (self.alpha / n_samples) * self.weights
                elif self.penalty == 'l1':
                    dw += (self.alpha / n_samples) * np.sign(self.weights)

                # Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        C : array of shape (n_samples,)
            Predicted values.
        """
        # Handle missing values in prediction data
        X, _ = self._handle_missing_values(X, y=None, fit=False)
        
        X = np.array(X, dtype=float)
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - (u / v)
