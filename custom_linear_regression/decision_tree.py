"""
Decision Tree Regressor
=======================
A simple NumPy-based implementation of a decision tree for regression.
"""

import numpy as np
from .exceptions import NotFittedError


class DecisionTreeRegressor:
    """
    A decision tree regressor.

    Parameters
    ----------
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least min_samples_leaf samples in each of the left and right branches.

    Features
    --------
    - Supports numerical features.
    - Uses mean squared error (MSE) as the splitting criterion.
    - Builds the tree using a greedy, top-down approach.
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def fit(self, X, y):
        """Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (real numbers).

        Returns
        -------
        self : object
            Returns self.
        """
        # Convert to numpy arrays
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # Store number of features
        self.n_features_ = X.shape[1]

        # Initialize and grow the tree
        self.tree_ = self._grow_tree(X, y)
        
        # Compute tree properties after fitting
        self.max_depth_ = self._get_tree_depth(self.tree_)
        self.n_leaves_ = self._count_leaves(self.tree_)
        self.feature_importances_ = self._compute_feature_importances(X.shape[1])
        
        return self

    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the decision tree.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        y : array-like, shape (n_samples,)
            The target values.

        depth : int, default=0
            The current depth of the node.

        Returns
        -------
        dict : A node in the decision tree.
            If the node is a leaf, it will have a 'value' key.
            If the node is internal, it will have:
                'feature_index': int, the index of the feature to split on.
                'threshold': float, the threshold value for the split.
                'left': dict, the left subtree.
                'right': dict, the right subtree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            leaf_value = self._calculate_leaf_value(y)
            return {"value": leaf_value}

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, n_features)
        if best_feature is None:
            leaf_value = self._calculate_leaf_value(y)
            return {"value": leaf_value}

        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # Check if split meets min_samples_leaf
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            leaf_value = self._calculate_leaf_value(y)
            return {"value": leaf_value}

        # Recursively grow the left and right subtrees
        left_subtree = self._grow_tree(X_left, y_left, depth + 1)
        right_subtree = self._grow_tree(X_right, y_right, depth + 1)

        return {
            "feature_index": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _best_split(self, X, y, n_features):
        """Find the best split for a node.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        y : array-like, shape (n_samples,)
            The target values.

        n_features : int
            The number of features.

        Returns
        -------
        tuple: (best_feature, best_threshold) or (None, None) if no split found.
        """
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature_index in range(n_features):
            # Get the unique values for the feature
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                # Split the data
                left_indices = feature_values <= threshold
                right_indices = feature_values > threshold

                if len(y[left_indices]) < self.min_samples_leaf or len(y[right_indices]) < self.min_samples_leaf:
                    continue

                # Calculate MSE for this split
                mse = self._calculate_mse(y[left_indices], y[right_indices])
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_mse(self, left_y, right_y):
        """Calculate the mean squared error for a split.

        Parameters
        ----------
        left_y : array-like, shape (n_left,)
            Target values for the left subset.

        right_y : array-like, shape (n_right,)
            Target values for the right subset.

        Returns
        -------
        float : The weighted MSE of the split.
        """
        n = len(left_y) + len(right_y)
        if n == 0:
            return 0
        mse_left = np.mean((left_y - np.mean(left_y)) ** 2) if len(left_y) > 0 else 0
        mse_right = np.mean((right_y - np.mean(right_y)) ** 2) if len(right_y) > 0 else 0
        return (len(left_y) * mse_left + len(right_y) * mse_right) / n

    def _calculate_leaf_value(self, y):
        """Calculate the value for a leaf node (mean of y)."""
        return np.mean(y)

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like, shape (n_samples,)
            The predicted values.
        """
        if self.tree_ is None:
            raise NotFittedError(
                "This DecisionTreeRegressor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        X = np.array(X, dtype=np.float64)
        return np.array([self._predict_tree(inputs, self.tree_) for inputs in X])

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.

        y : array-like, shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt y.
        """
        y_pred = self.predict(X)
        # Calculate R^2: 1 - (sum of squared residuals) / (total sum of squares)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 0.0  # Avoid division by zero; if all y are same, R^2 is 0 (or undefined, but we return 0)
        return 1 - ss_res / ss_tot

    def _predict_tree(self, inputs, tree):
        """Predict a single sample by traversing the tree.

        Parameters
        ----------
        inputs : array-like, shape (n_features,)
            The input sample.

        tree : dict
            The decision tree (or subtree) to traverse.

        Returns
        -------
        float : The predicted value.
        """
        if "value" in tree:
            return tree["value"]

        feature_index = tree["feature_index"]
        threshold = tree["threshold"]
        if inputs[feature_index] <= threshold:
            return self._predict_tree(inputs, tree["left"])
        else:
            return self._predict_tree(inputs, tree["right"])

    def _get_tree_depth(self, tree):
        """Calculate the maximum depth of the tree.
        
        Parameters
        ----------
        tree : dict
            The decision tree (or subtree).
            
        Returns
        -------
        int : The depth of the tree.
        """
        if "value" in tree:
            return 0
        left_depth = self._get_tree_depth(tree["left"])
        right_depth = self._get_tree_depth(tree["right"])
        return max(left_depth, right_depth) + 1

    def _count_leaves(self, tree):
        """Count the number of leaf nodes in the tree.
        
        Parameters
        ----------
        tree : dict
            The decision tree (or subtree).
            
        Returns
        -------
        int : The number of leaf nodes.
        """
        if "value" in tree:
            return 1
        return self._count_leaves(tree["left"]) + self._count_leaves(tree["right"])

    def _compute_feature_importances(self, n_features):
        """Compute feature importance based on total reduction in MSE.
        
        Parameters
        ----------
        n_features : int
            Number of features.
            
        Returns
        -------
        array : Feature importances (sum to 1).
        """
        importances = np.zeros(n_features)
        self._accumulate_feature_importance(self.tree_, importances)
        
        # Normalize to sum to 1
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        return importances

    def _accumulate_feature_importance(self, tree, importances):
        """Recursively accumulate feature importance from the tree.
        
        Parameters
        ----------
        tree : dict
            The decision tree (or subtree).
        importances : array
            Array to accumulate importances into.
        """
        if "value" not in tree:
            feature_index = tree["feature_index"]
            # Calculate the weight of this split (proportion of samples)
            # For simplicity, we'll use equal weighting - in a full implementation,
            # this would be weighted by the number of samples at this node
            importances[feature_index] += 1.0
            self._accumulate_feature_importance(tree["left"], importances)
            self._accumulate_feature_importance(tree["right"], importances)
