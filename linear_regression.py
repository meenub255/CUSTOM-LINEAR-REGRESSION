import numpy as np

class LinearRegression:
    """
    Linear Regression with optional L1 (Lasso) and L2 (Ridge) regularization.
    Implemented purely using NumPy.
    """
    def __init__(self, penalty=None, alpha=1.0, lr=0.01, n_iters=1000, solver='gd', loss='mse'):
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

        self.penalty = penalty
        self.alpha = alpha
        self.lr = lr
        self.n_iters = n_iters
        self.solver = solver
        self.loss = loss
        
        self.weights = None
        self.bias = None

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
