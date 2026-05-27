"""
Principal Component Analysis (PCA)
====================================
A simple NumPy-based implementation of PCA for dimensionality reduction.
"""

import numpy as np
import warnings

from .exceptions import NotFittedError


class PCA:
    """
    Principal Component Analysis (PCA)

    Linear dimensionality reduction using Singular Value Decomposition (SVD)
    of the data to project it to a lower dimensional space.

    Parameters
    ----------
    n_components : int, float, None, default=None
        Number of components to keep.
        If n_components is not set, all components are kept:
        n_components == min(n_samples, n_features).
        If n_components is a float between 0 and 1, it represents the
        proportion of variance to keep.

    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto:
            The solver is selected by a default policy based on X.shape and
            n_components: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated to the desired number of components.
        full: run exact full SVD calling the standard LAPACK solver via
            scipy.svd and select the components by postprocessing
        arpack: run SVD truncated to n_components calling ARPACK solver via
            scipy.sparse.linalg.svds. It requires strictly
            0 < n_components < min(X.shape).
        randomized: run randomized SVD by the method of Halko et al.

    whiten : bool, default=False
        When True (False by default) the components_ are multiplied by the
        square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

    random_state : int, RandomState instance or None, default=None
        Used when svd_solver=='randomized'. Pass an int for reproducible
        results across multiple function calls.

    Notes
    -----
    This implementation centers the data but does not scale it.
    For scaled PCA, standardize the data before calling fit.
    """

    def __init__(
        self,
        n_components=None,
        svd_solver="auto",
        whiten=False,
        random_state=None,
    ):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Convert to numpy array
        X = np.array(X, dtype=np.float64, copy=False)

        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Determine the number of components
        n_samples, n_features = X.shape

        if self.n_components is None:
            n_components = min(n_samples, n_features)
        elif isinstance(self.n_components, float):
            # n_components is a proportion of variance
            # We'll compute the full SVD first to determine n_components
            # based on explained variance ratio
            n_components = min(n_samples, n_features)
        else:
            n_components = int(self.n_components)
            if not 0 < n_components <= min(n_samples, n_features):
                raise ValueError(
                    "n_components must be between 0 and min(n_samples, n_features)"
                )

        # Perform SVD
        # We'll use numpy.linalg.svd for simplicity (full SVD)
        # For large datasets, we might want to use randomized or truncated,
        # but for now we'll use full SVD and then truncate.
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # The components are the rows of Vt
        components = Vt

        # Explained variance: squared singular values / (n_samples - 1)
        explained_variance_ = (S**2) / (n_samples - 1)
        total_variance = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_variance

        # If n_components is a float, we need to select the number of components
        # that explain at least that proportion of variance
        if isinstance(self.n_components, float):
            # Compute cumulative explained variance ratio
            cumsum = np.cumsum(explained_variance_ratio_)
            n_components = np.argmax(cumsum >= self.n_components) + 1
            # Ensure at least one component
            n_components = max(1, n_components)

        # Store the components (only the first n_components)
        self.components_ = components[:n_components]
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = S[:n_components]

        # Whitening
        if self.whiten:
            self.components_ /= np.sqrt(self.explained_variance_)[:, np.newaxis]

        self.n_components_ = n_components
        return self

    def transform(self, X):
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed values.
        """
        # Check if fit has been called
        if not hasattr(self, "components_"):
            raise NotFittedError(
                "This PCA instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )

        X = np.array(X, dtype=np.float64, copy=False)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed values.
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original : array-like, shape (n_samples, n_features)
            Reconstructed data.
        """
        if not hasattr(self, "components_"):
            raise NotFittedError(
                "This PCA instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )

        return np.dot(X, self.components_) + self.mean_