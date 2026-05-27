"""
K-Nearest Neighbours Regression
================================
Pure-NumPy implementation that mirrors the API of ``LinearRegression`` in
this package:

* ``fit(X, y)``
* ``predict(X)``
* ``score(X, y)``           → R²
* ``feature_importances_``  → permutation-based importance array

Supported options
-----------------
metric      : 'euclidean' | 'manhattan' | 'minkowski'
weights     : 'uniform'   | 'distance'   (inverse-distance weighted average)
p           : Minkowski exponent (only used when metric='minkowski')
scale       : bool – z-score feature scaling before distance computation
              (strongly recommended; default True)
k           : number of neighbours (default 5)

Missing-value and outlier handling use the same strategies as LinearRegression.
"""

import numpy as np
import warnings

from .exceptions import NotFittedError, DataQualityError


# ---------------------------------------------------------------------------
# Helper distance functions (pure NumPy, no scipy dependency)
# ---------------------------------------------------------------------------

def _euclidean_distances(X_query, X_train):
    """
    Compute pairwise Euclidean distances.

    Parameters
    ----------
    X_query : ndarray, shape (n_query, n_features)
    X_train : ndarray, shape (n_train, n_features)

    Returns
    -------
    distances : ndarray, shape (n_query, n_train)
    """
    # ||a - b||² = ||a||² + ||b||² - 2·a·bᵀ  — numerically stable for large arrays
    sq_query = np.sum(X_query ** 2, axis=1, keepdims=True)   # (n_query, 1)
    sq_train = np.sum(X_train ** 2, axis=1, keepdims=True).T  # (1, n_train)
    cross    = X_query @ X_train.T                             # (n_query, n_train)
    sq_dist  = sq_query + sq_train - 2.0 * cross
    # Numerical noise can give tiny negatives → clip
    sq_dist  = np.maximum(sq_dist, 0.0)
    return np.sqrt(sq_dist)


def _manhattan_distances(X_query, X_train):
    """Pairwise L1 distances — computed row-by-row to avoid huge memory spike."""
    n_query = X_query.shape[0]
    n_train = X_train.shape[0]
    dist    = np.empty((n_query, n_train), dtype=float)
    for i in range(n_query):
        dist[i] = np.sum(np.abs(X_query[i] - X_train), axis=1)
    return dist


def _minkowski_distances(X_query, X_train, p):
    """Pairwise Minkowski distances with exponent *p*."""
    n_query = X_query.shape[0]
    n_train = X_train.shape[0]
    dist    = np.empty((n_query, n_train), dtype=float)
    for i in range(n_query):
        dist[i] = np.sum(np.abs(X_query[i] - X_train) ** p, axis=1) ** (1.0 / p)
    return dist


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class KNNRegression:
    """
    K-Nearest Neighbours Regression (pure NumPy).

    Parameters
    ----------
    k : int
        Number of neighbours to use. Default: 5.
    metric : str
        Distance metric. One of ``'euclidean'``, ``'manhattan'``,
        ``'minkowski'``. Default: ``'euclidean'``.
    weights : str
        Neighbour weighting strategy.

        * ``'uniform'``  – simple average of the *k* neighbours' targets.
        * ``'distance'`` – inverse-distance weighted average; closer
          neighbours contribute more. Exact ties get equal weight.

        Default: ``'uniform'``.
    p : float
        Exponent for the Minkowski metric (ignored for other metrics).
        ``p=1`` is Manhattan, ``p=2`` is Euclidean. Default: 3.
    scale : bool
        If ``True`` (default), features are z-score scaled before distance
        computation using statistics from the training set.  Strongly
        recommended when features are on different scales.
    missing_strategy : str
        Strategy for imputing NaN values.  One of ``'mean'``,
        ``'median'``, ``'forward_fill'``, ``'backward_fill'``.
        Default: ``'mean'``.
    drop_missing : bool
        If ``True``, rows with missing values are *dropped* during training
        (rather than imputed).  Prediction always imputes to keep the sample
        count stable. Default: ``False``.
    outlier_strategy : str or None
        Optional outlier handling. One of ``None``, ``'zscore'``, ``'iqr'``.
        Default: ``None``.
    outlier_threshold : float
        Threshold for the outlier strategy (z-score cutoff or IQR
        multiplier). Default: 3.0.
    outlier_action : str
        How to handle detected outliers during training.  ``'remove'``
        drops the rows; ``'clip'`` winsorises them. Default: ``'remove'``.

    Attributes
    ----------
    X_train_ : ndarray, shape (n_samples, n_features)
        Training feature matrix (stored after ``fit``).
    y_train_ : ndarray, shape (n_samples,)
        Training target vector (stored after ``fit``).
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_importances_ : ndarray, shape (n_features,)
        Permutation-based importance scores (available after calling
        ``compute_feature_importances``).

    Examples
    --------
    >>> import numpy as np
    >>> from custom_linear_regression import KNNRegression
    >>> rng = np.random.default_rng(0)
    >>> X_train = rng.random((100, 3))
    >>> y_train = X_train @ np.array([2, -1, 0.5]) + rng.normal(0, 0.1, 100)
    >>> X_test  = rng.random((20, 3))
    >>> y_test  = X_test  @ np.array([2, -1, 0.5]) + rng.normal(0, 0.1, 20)
    >>> model = KNNRegression(k=5, metric='euclidean', weights='distance')
    >>> model.fit(X_train, y_train)
    KNNRegression(k=5, metric='euclidean', weights='distance')
    >>> print(f"R² = {model.score(X_test, y_test):.4f}")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        k=5,
        metric="euclidean",
        weights="uniform",
        p=3,
        scale=True,
        missing_strategy="mean",
        drop_missing=False,
        outlier_strategy=None,
        outlier_threshold=3.0,
        outlier_action="remove",
    ):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if metric not in ("euclidean", "manhattan", "minkowski"):
            raise ValueError("metric must be 'euclidean', 'manhattan', or 'minkowski'.")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'.")
        if p <= 0:
            raise ValueError("p must be positive.")
        if missing_strategy not in ("mean", "median", "forward_fill", "backward_fill"):
            raise ValueError(
                "missing_strategy must be 'mean', 'median', 'forward_fill', or 'backward_fill'."
            )
        if outlier_strategy not in (None, "zscore", "iqr"):
            raise ValueError("outlier_strategy must be None, 'zscore', or 'iqr'.")
        if outlier_action not in ("remove", "clip"):
            raise ValueError("outlier_action must be 'remove' or 'clip'.")

        self.k                 = k
        self.metric            = metric
        self.weights           = weights
        self.p                 = p
        self.scale             = scale
        self.missing_strategy  = missing_strategy
        self.drop_missing      = drop_missing
        self.outlier_strategy  = outlier_strategy
        self.outlier_threshold = outlier_threshold
        self.outlier_action    = outlier_action

        # Set after fit
        self.X_train_          = None
        self.y_train_          = None
        self.n_features_in_    = None
        self.feature_importances_ = None

        # Scaling statistics (computed from training data)
        self._scale_mean       = None
        self._scale_std        = None

        # Missing-value statistics
        self._feature_means    = None
        self._feature_medians  = None

        # Outlier bounds
        self._feat_lower       = None
        self._feat_upper       = None
        self._tgt_lower        = None
        self._tgt_upper        = None

    # ------------------------------------------------------------------
    # Missing-value helpers  (mirrors LinearRegression logic)
    # ------------------------------------------------------------------

    def _compute_feature_statistics(self, X):
        means   = np.zeros(X.shape[1], dtype=float)
        medians = np.zeros(X.shape[1], dtype=float)
        for col in range(X.shape[1]):
            valid = X[~np.isnan(X[:, col]), col]
            if valid.size == 0:
                means[col] = medians[col] = 0.0
            else:
                means[col]   = np.mean(valid)
                medians[col] = np.median(valid)
        return means, medians

    def _impute(self, X):
        X_out = X.copy()
        for col in range(X_out.shape[1]):
            mask = np.isnan(X_out[:, col])
            if not mask.any():
                continue
            if self.missing_strategy == "mean":
                X_out[mask, col] = self._feature_means[col]
            elif self.missing_strategy == "median":
                X_out[mask, col] = self._feature_medians[col]
            elif self.missing_strategy == "forward_fill":
                for row in range(X_out.shape[0]):
                    if np.isnan(X_out[row, col]):
                        prev = X_out[row - 1, col] if row > 0 else np.nan
                        X_out[row, col] = prev if not np.isnan(prev) else self._feature_means[col]
            elif self.missing_strategy == "backward_fill":
                for row in range(X_out.shape[0] - 1, -1, -1):
                    if np.isnan(X_out[row, col]):
                        nxt = X_out[row + 1, col] if row < X_out.shape[0] - 1 else np.nan
                        X_out[row, col] = nxt if not np.isnan(nxt) else self._feature_means[col]
        return X_out

    def _handle_missing(self, X, y=None, fit=False):
        X = np.array(X, dtype=float)
        y_arr = None if y is None else np.array(y, dtype=float)

        n_missing = int(np.isnan(X).sum())
        if n_missing > 0:
            warnings.warn(
                f"Found {n_missing} missing values in X. "
                f"Using '{self.missing_strategy}' imputation.",
                stacklevel=3,
            )

        if fit:
            self._feature_means, self._feature_medians = self._compute_feature_statistics(X)

        if self._feature_means is None:
            raise ValueError("Missing-value statistics not initialised — call fit() first.")

        if self.drop_missing and y_arr is not None:
            mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_arr)
            X_clean, y_clean = X[mask], y_arr[mask]
            if X_clean.size == 0:
                raise DataQualityError("All rows dropped when handling missing values.")
            return X_clean, y_clean

        if self.drop_missing and y_arr is None and n_missing > 0:
            warnings.warn(
                "drop_missing=True only applies during training; "
                "prediction data is imputed to preserve sample count.",
                stacklevel=3,
            )

        X_clean = self._impute(X)

        if y_arr is None:
            return X_clean, None

        valid = ~np.isnan(y_arr)
        X_clean, y_clean = X_clean[valid], y_arr[valid]
        if X_clean.size == 0:
            raise DataQualityError("All target values are missing.")
        return X_clean, y_clean

    # ------------------------------------------------------------------
    # Outlier helpers  (mirrors LinearRegression logic)
    # ------------------------------------------------------------------

    def _outlier_bounds(self, values):
        if self.outlier_strategy == "zscore":
            mu  = np.mean(values, axis=0)
            sig = np.std(values,  axis=0)
            sig = np.where(sig == 0, 1.0, sig)
            return mu - self.outlier_threshold * sig, mu + self.outlier_threshold * sig
        else:  # iqr
            q1  = np.percentile(values, 25, axis=0)
            q3  = np.percentile(values, 75, axis=0)
            iqr = q3 - q1
            return q1 - self.outlier_threshold * iqr, q3 + self.outlier_threshold * iqr

    def _handle_outliers(self, X, y=None, fit=False):
        X     = np.array(X, dtype=float)
        y_arr = None if y is None else np.array(y, dtype=float)

        if self.outlier_strategy is None:
            return X, y_arr

        if fit:
            self._feat_lower, self._feat_upper = self._outlier_bounds(X)
            if y_arr is not None:
                yl, yu         = self._outlier_bounds(y_arr.reshape(-1, 1))
                self._tgt_lower = float(yl[0])
                self._tgt_upper = float(yu[0])

        if self._feat_lower is None:
            raise ValueError("Outlier bounds not initialised — call fit() first.")

        prediction_mode = y_arr is None
        action = self.outlier_action
        if prediction_mode and action == "remove":
            warnings.warn(
                "Outlier removal only applies during training; "
                "prediction data is clipped.",
                stacklevel=3,
            )
            action = "clip"

        if action == "clip":
            X_proc = np.clip(X, self._feat_lower, self._feat_upper)
            if y_arr is None:
                return X_proc, None
            y_proc = y_arr
            if self._tgt_lower is not None:
                y_proc = np.clip(y_arr, self._tgt_lower, self._tgt_upper)
            return X_proc, y_proc

        # action == "remove"
        mask = ((X >= self._feat_lower) & (X <= self._feat_upper)).all(axis=1)
        if y_arr is not None and self._tgt_lower is not None:
            mask &= (y_arr >= self._tgt_lower) & (y_arr <= self._tgt_upper)

        removed = int((~mask).sum())
        if removed > 0:
            warnings.warn(
                f"Removed {removed} outlier rows using '{self.outlier_strategy}' strategy.",
                stacklevel=3,
            )

        X_proc = X[mask]
        if y_arr is None:
            return X_proc, None
        y_proc = y_arr[mask]
        if X_proc.size == 0:
            raise DataQualityError("All rows removed as outliers.")
        return X_proc, y_proc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale(self, X):
        """Apply training-set z-score scaling."""
        return (X - self._scale_mean) / self._scale_std

    def _compute_distances(self, X_query, X_train):
        """Return distance matrix (n_query × n_train) for the chosen metric."""
        if self.metric == "euclidean":
            return _euclidean_distances(X_query, X_train)
        elif self.metric == "manhattan":
            return _manhattan_distances(X_query, X_train)
        else:  # minkowski
            return _minkowski_distances(X_query, X_train, self.p)

    def _validate_fitted(self):
        if self.X_train_ is None or self.y_train_ is None:
            raise NotFittedError(
                "This KNNRegression instance is not fitted yet. Call fit() first."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Store the (pre-processed) training data.

        KNN is a lazy learner — no explicit model is built; training data is
        kept in memory and used at prediction time.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = self._handle_missing(X, y, fit=True)
        X, y = self._handle_outliers(X, y, fit=True)

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        self.n_features_in_ = X.shape[1]

        if self.scale:
            self._scale_mean = np.mean(X, axis=0)
            self._scale_std  = np.std(X,  axis=0)
            # Avoid division by zero for constant features
            self._scale_std  = np.where(self._scale_std == 0, 1.0, self._scale_std)
            self.X_train_    = self._scale(X)
        else:
            self._scale_mean = np.zeros(X.shape[1])
            self._scale_std  = np.ones(X.shape[1])
            self.X_train_    = X.copy()

        self.y_train_ = y.copy()
        return self

    def predict(self, X):
        """
        Predict target values for *X*.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
        """
        self._validate_fitted()

        X, _ = self._handle_missing(X, fit=False)
        X, _ = self._handle_outliers(X, fit=False)
        X     = np.array(X, dtype=float)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was trained on "
                f"{self.n_features_in_} features."
            )

        X_scaled = self._scale(X)

        # Effective k cannot exceed the number of training samples
        k_eff = min(self.k, self.X_train_.shape[0])

        dist_matrix = self._compute_distances(X_scaled, self.X_train_)
        # Indices of the k nearest neighbours for each query point
        nn_indices  = np.argpartition(dist_matrix, k_eff - 1, axis=1)[:, :k_eff]

        y_pred = np.empty(X.shape[0], dtype=float)

        for i in range(X.shape[0]):
            idx   = nn_indices[i]
            dists = dist_matrix[i, idx]
            targets = self.y_train_[idx]

            if self.weights == "uniform":
                y_pred[i] = np.mean(targets)
            else:  # distance weighting
                # Guard against exact-zero distances (query == training point)
                zero_mask = dists == 0.0
                if zero_mask.any():
                    # If any neighbour is exactly the query point, just average those
                    y_pred[i] = np.mean(targets[zero_mask])
                else:
                    w = 1.0 / dists
                    y_pred[i] = np.dot(w, targets) / w.sum()

        return y_pred

    def score(self, X, y):
        """
        Return the coefficient of determination R² of the prediction.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        r2 : float
        """
        y      = np.array(y, dtype=float)
        y_pred = self.predict(X)

        if y.shape[0] != y_pred.shape[0]:
            raise ValueError("Prediction and target lengths do not match.")

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0.0:
            # All targets are identical — return 1.0 if perfect fit, else 0.0
            return 1.0 if ss_res == 0.0 else 0.0

        return float(1.0 - ss_res / ss_tot)

    def compute_feature_importances(self, X, y, n_repeats=5, random_state=None):
        """
        Estimate feature importances via **permutation importance**.

        For each feature, the column is randomly shuffled *n_repeats* times
        and the average drop in R² is recorded.  Features whose shuffling
        causes a large performance drop are deemed more important.

        The result is stored in ``self.feature_importances_`` and also
        returned for convenience.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Validation/test set features.
        y : array-like, shape (n_samples,)
            Validation/test set targets.
        n_repeats : int
            Number of shuffle repetitions per feature. Default: 5.
        random_state : int or None
            Seed for reproducibility.

        Returns
        -------
        importances : ndarray, shape (n_features,)
            Mean R² drop when each feature is permuted.
        """
        self._validate_fitted()

        rng      = np.random.default_rng(random_state)
        baseline = self.score(X, y)
        X_arr    = np.array(X, dtype=float)
        importances = np.zeros(self.n_features_in_, dtype=float)

        for feat_idx in range(self.n_features_in_):
            drops = np.empty(n_repeats, dtype=float)
            for rep in range(n_repeats):
                X_permuted = X_arr.copy()
                X_permuted[:, feat_idx] = rng.permutation(X_permuted[:, feat_idx])
                drops[rep] = baseline - self.score(X_permuted, y)
            importances[feat_idx] = np.mean(drops)

        self.feature_importances_ = importances
        return importances

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"KNNRegression(k={self.k}, metric='{self.metric}', "
            f"weights='{self.weights}')"
        )

    def get_params(self):
        """Return constructor parameters as a dict."""
        return {
            "k":                 self.k,
            "metric":            self.metric,
            "weights":           self.weights,
            "p":                 self.p,
            "scale":             self.scale,
            "missing_strategy":  self.missing_strategy,
            "drop_missing":      self.drop_missing,
            "outlier_strategy":  self.outlier_strategy,
            "outlier_threshold": self.outlier_threshold,
            "outlier_action":    self.outlier_action,
        }
