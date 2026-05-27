import numpy as np
import warnings

from .exceptions import NotFittedError, OptimizationFailedError, DataQualityError


class LinearRegression:
    """
    Linear Regression with optional L1 (Lasso) and L2 (Ridge) regularization.
    Implemented purely using NumPy.
    """

    def __init__(
        self,
        fit_intercept=True,
        penalty=None,
        alpha=1.0,
        lr=0.01,
        n_iters=1000,
        solver="gd",
        loss="mse",
        missing_strategy="mean",
        drop_missing=False,
        outlier_strategy=None,
        outlier_threshold=3.0,
        outlier_action="remove",
    ):
        """
        Parameters:
        -----------
        fit_intercept : bool
            Whether to calculate the intercept for this model.
        penalty : str or None
            Regularization penalty. Can be None (OLS), 'l1' (Lasso), or 'l2' (Ridge).
        alpha : float
            Regularization strength.
        lr : float
            Learning rate for Gradient Descent solver.
        n_iters : int
            Number of iterations for Gradient Descent solver.
        solver : str
            'gd' (Gradient Descent) or 'closed' (Closed-form solution).
            Note: 'closed' is only available for None and 'l2' penalty.
        loss : str
            Loss function to optimize. Can be 'mse' (Mean Squared Error) or 'mae' (Mean Absolute Error).
            Note: 'closed' solver only supports 'mse'.
        missing_strategy : str
            Strategy for handling missing values. Options: 'mean', 'median', 'forward_fill', 'backward_fill'.
            Default: 'mean'.
        drop_missing : bool
            If True, drop rows with missing values during training.
            Prediction keeps sample count stable by imputing instead of dropping.
        outlier_strategy : str or None
            Optional outlier handling strategy. Can be None, 'zscore', or 'iqr'.
        outlier_threshold : float
            Threshold used by the selected outlier strategy.
            For z-score this is the z cutoff; for IQR this is the multiplier.
        outlier_action : str
            How to handle outliers during training. Can be 'remove' or 'clip'.
        """
        if penalty not in [None, "l1", "l2"]:
            raise ValueError("Penalty must be None, 'l1', or 'l2'")
        if solver not in ["gd", "closed"]:
            raise ValueError("Solver must be 'gd' or 'closed'")
        if solver == "closed" and penalty == "l1":
            raise ValueError("Closed-form solution is not available for L1 penalty (Lasso). Use solver='gd'.")
        if loss not in ["mse", "mae"]:
            raise ValueError("Loss must be 'mse' or 'mae'")
        if solver == "closed" and loss != "mse":
            raise ValueError("Closed-form solver only supports 'mse' loss.")
        if missing_strategy not in ["mean", "median", "forward_fill", "backward_fill"]:
            raise ValueError("Missing strategy must be 'mean', 'median', 'forward_fill', or 'backward_fill'")
        if outlier_strategy not in [None, "zscore", "iqr"]:
            raise ValueError("Outlier strategy must be None, 'zscore', or 'iqr'")
        if outlier_action not in ["remove", "clip"]:
            raise ValueError("Outlier action must be 'remove' or 'clip'")

        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.alpha = alpha
        self.lr = lr
        self.n_iters = n_iters
        self.solver = solver
        self.loss = loss
        self.missing_strategy = missing_strategy
        self.drop_missing = drop_missing
        self.outlier_strategy = outlier_strategy
        self.outlier_threshold = outlier_threshold
        self.outlier_action = outlier_action

        self.weights = None
        self.bias = None
        self.feature_means = None
        self.feature_medians = None
        self.feature_lower_bounds = None
        self.feature_upper_bounds = None
        self.target_lower_bound = None
        self.target_upper_bound = None

    def _compute_feature_statistics(self, X):
        feature_means = np.zeros(X.shape[1], dtype=float)
        feature_medians = np.zeros(X.shape[1], dtype=float)

        for col in range(X.shape[1]):
            valid_values = X[~np.isnan(X[:, col]), col]
            if valid_values.size == 0:
                feature_means[col] = 0.0
                feature_medians[col] = 0.0
            else:
                feature_means[col] = np.mean(valid_values)
                feature_medians[col] = np.median(valid_values)

        return feature_means, feature_medians

    def _impute_missing_features(self, X):
        X_imputed = X.copy()

        if self.missing_strategy == "mean":
            fill_values = self.feature_means
            for col in range(X_imputed.shape[1]):
                mask = np.isnan(X_imputed[:, col])
                if mask.any():
                    X_imputed[mask, col] = fill_values[col]

        elif self.missing_strategy == "median":
            fill_values = self.feature_medians
            for col in range(X_imputed.shape[1]):
                mask = np.isnan(X_imputed[:, col])
                if mask.any():
                    X_imputed[mask, col] = fill_values[col]

        elif self.missing_strategy == "forward_fill":
            for col in range(X_imputed.shape[1]):
                for row in range(X_imputed.shape[0]):
                    if np.isnan(X_imputed[row, col]):
                        if row > 0 and not np.isnan(X_imputed[row - 1, col]):
                            X_imputed[row, col] = X_imputed[row - 1, col]
                        else:
                            X_imputed[row, col] = self.feature_means[col]

        elif self.missing_strategy == "backward_fill":
            for col in range(X_imputed.shape[1]):
                for row in range(X_imputed.shape[0] - 1, -1, -1):
                    if np.isnan(X_imputed[row, col]):
                        if row < X_imputed.shape[0] - 1 and not np.isnan(X_imputed[row + 1, col]):
                            X_imputed[row, col] = X_imputed[row + 1, col]
                        else:
                            X_imputed[row, col] = self.feature_means[col]

        return X_imputed

    def _handle_missing_values(self, X, y=None, fit=False):
        """
        Handle missing values in features and target.
        """
        X = np.array(X, dtype=float)
        y_array = None if y is None else np.array(y, dtype=float)

        n_missing = np.isnan(X).sum()
        if n_missing > 0:
            warnings.warn(
                f"Found {n_missing} missing values in X. Using {self.missing_strategy} handling.",
                stacklevel=2,
            )

        if fit:
            self.feature_means, self.feature_medians = self._compute_feature_statistics(X)

        if self.feature_means is None or self.feature_medians is None:
            raise ValueError("Missing-value statistics are not initialized. Fit the model first.")

        if self.drop_missing and y_array is not None:
            valid_mask = ~np.isnan(X).any(axis=1)
            valid_mask &= ~np.isnan(y_array)
            X_clean = X[valid_mask]
            y_clean = y_array[valid_mask]
            if X_clean.size == 0:
                raise DataQualityError("All rows were dropped while handling missing values.")
            return X_clean, y_clean

        if self.drop_missing and y_array is None and n_missing > 0:
            warnings.warn(
                "drop_missing=True is only applied during training. Prediction data is imputed to preserve sample count.",
                stacklevel=2,
            )

        X_clean = self._impute_missing_features(X)

        if y_array is None:
            return X_clean, None

        valid_mask = ~np.isnan(y_array)
        X_clean = X_clean[valid_mask]
        y_clean = y_array[valid_mask]
        if X_clean.size == 0:
            raise DataQualityError("All target rows were dropped because y contains only missing values.")
        return X_clean, y_clean

    def _compute_outlier_bounds(self, values):
        values = np.array(values, dtype=float)

        if self.outlier_strategy == "zscore":
            centers = np.mean(values, axis=0)
            spreads = np.std(values, axis=0)
            spreads = np.where(spreads == 0, 1.0, spreads)
            lower = centers - self.outlier_threshold * spreads
            upper = centers + self.outlier_threshold * spreads
        else:
            q1 = np.percentile(values, 25, axis=0)
            q3 = np.percentile(values, 75, axis=0)
            iqr = q3 - q1
            lower = q1 - self.outlier_threshold * iqr
            upper = q3 + self.outlier_threshold * iqr

        return lower, upper

    def _handle_outliers(self, X, y=None, fit=False):
        """
        Handle outliers in features and optionally target values.
        """
        X = np.array(X, dtype=float)
        y_array = None if y is None else np.array(y, dtype=float)

        if self.outlier_strategy is None:
            return X, y_array

        if fit:
            self.feature_lower_bounds, self.feature_upper_bounds = self._compute_outlier_bounds(X)
            if y_array is not None:
                y_lower, y_upper = self._compute_outlier_bounds(y_array.reshape(-1, 1))
                self.target_lower_bound = float(y_lower[0])
                self.target_upper_bound = float(y_upper[0])

        if self.feature_lower_bounds is None or self.feature_upper_bounds is None:
            raise ValueError("Outlier bounds are not initialized. Fit the model first.")

        prediction_mode = y_array is None
        action = self.outlier_action
        if prediction_mode and self.outlier_action == "remove":
            warnings.warn(
                "Outlier removal is only applied during training. Prediction data is clipped to preserve sample count.",
                stacklevel=2,
            )
            action = "clip"

        if action == "clip":
            X_processed = np.clip(X, self.feature_lower_bounds, self.feature_upper_bounds)
            if y_array is None:
                return X_processed, None
            if self.target_lower_bound is None or self.target_upper_bound is None:
                return X_processed, y_array
            y_processed = np.clip(y_array, self.target_lower_bound, self.target_upper_bound)
            return X_processed, y_processed

        row_mask = ((X >= self.feature_lower_bounds) & (X <= self.feature_upper_bounds)).all(axis=1)
        if y_array is not None and self.target_lower_bound is not None and self.target_upper_bound is not None:
            row_mask &= (y_array >= self.target_lower_bound) & (y_array <= self.target_upper_bound)

        removed_rows = int((~row_mask).sum())
        if removed_rows > 0:
            warnings.warn(
                f"Removed {removed_rows} outlier rows using {self.outlier_strategy} strategy.",
                stacklevel=2,
            )

        X_processed = X[row_mask]
        if y_array is None:
            return X_processed, None
        y_processed = y_array[row_mask]
        if X_processed.size == 0:
            raise DataQualityError("All rows were removed while handling outliers.")
        return X_processed, y_processed

    def _fit_gradient_descent(self, X, y):
        n_samples, n_features = X.shape

        feature_means = np.mean(X, axis=0)
        feature_scales = np.std(X, axis=0)
        feature_scales = np.where(feature_scales == 0, 1.0, feature_scales)
        X_scaled = (X - feature_means) / feature_scales

        weights_scaled = np.zeros(n_features, dtype=float)
        bias_scaled = 0.0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X_scaled, weights_scaled) + bias_scaled
            errors = y_predicted - y

            if self.loss == "mse":
                dw = np.dot(X_scaled.T, errors) / n_samples
                db = np.mean(errors) if self.fit_intercept else 0.0
            else:
                signed_errors = np.sign(errors)
                dw = np.dot(X_scaled.T, signed_errors) / n_samples
                db = np.mean(signed_errors) if self.fit_intercept else 0.0

            if self.penalty == "l2":
                dw += (self.alpha / n_samples) * weights_scaled
                weights_scaled -= self.lr * dw
            elif self.penalty == "l1":
                weights_scaled -= self.lr * dw
                shrink = self.lr * self.alpha / n_samples
                weights_scaled = np.sign(weights_scaled) * np.maximum(np.abs(weights_scaled) - shrink, 0.0)
            else:
                weights_scaled -= self.lr * dw

            if self.fit_intercept:
                bias_scaled -= self.lr * db

            if not np.all(np.isfinite(weights_scaled)) or not np.isfinite(bias_scaled):
                raise OptimizationFailedError(
                    "Gradient descent diverged. Try a smaller learning rate or fewer iterations."
                )

        self.weights = weights_scaled / feature_scales
        if self.fit_intercept:
            self.bias = bias_scaled - np.dot(feature_means / feature_scales, weights_scaled)
        else:
            self.bias = 0.0

    def fit(self, X, y):
        """
        Fit the linear regression model.
        """
        X, y = self._handle_missing_values(X, y, fit=True)
        X, y = self._handle_outliers(X, y, fit=True)

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape

        if self.solver == "closed":
            if self.fit_intercept:
                X_b = np.c_[np.ones((n_samples, 1)), X]
                if self.penalty is None:
                    theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
                else:
                    identity = np.eye(X_b.shape[1])
                    identity[0, 0] = 0
                    theta = np.linalg.pinv(X_b.T.dot(X_b) + self.alpha * identity).dot(X_b.T).dot(y)
                self.bias = theta[0]
                self.weights = theta[1:]
            else:
                if self.penalty is None:
                    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
                else:
                    identity = np.eye(X.shape[1])
                    theta = np.linalg.pinv(X.T.dot(X) + self.alpha * identity).dot(X.T).dot(y)
                self.bias = 0.0
                self.weights = theta
        else:
            self._fit_gradient_descent(X, y)

        return self

    def predict(self, X):
        """
        Predict using the linear model.
        """
        self._validate_is_fitted()
        X, _ = self._handle_missing_values(X, y=None, fit=False)
        X, _ = self._handle_outliers(X, y=None, fit=False)
        X = np.array(X, dtype=float)
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.
        """
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        if y.shape[0] != y_pred.shape[0]:
            raise ValueError("Prediction and target lengths do not match.")
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - (u / v)

    @property
    def coef_(self):
        self._validate_is_fitted()
        return self.weights

    @property
    def intercept_(self):
        self._validate_is_fitted()
        return self.bias

    def _validate_is_fitted(self):
        if self.weights is None or self.bias is None:
            raise NotFittedError("This LinearRegression instance is not fitted yet.")