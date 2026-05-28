"""
Perceptron Classifier
=====================
A simple NumPy-based implementation of the perceptron algorithm for binary classification.
"""

import numpy as np
from .exceptions import NotFittedError


class PerceptronClassifier:
    """
    Perceptron classifier.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate (between 0.0 and 1.0).

    n_epochs : int, default=1000
        Number of passes over the training dataset.
        Note: The algorithm terminates early if no updates are made in an epoch.

    random_state : int, RandomState instance or None, default=None
        Seed for random number generator for weight initialization.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    weights_ : array, shape = [n_features]
        Weights after fitting.

    bias_ : float
        Bias term after fitting.

    errors_ : list
        Number of misclassifications in each epoch.

    Notes
    -----
    Implements the perceptron learning rule for binary classification.
    Assumes classes are labeled as -1 and 1.
    If your classes are 0 and 1, they will be converted internally.
    """

    def __init__(self, learning_rate=0.01, n_epochs=1000, random_state=None):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples,]
            Target values (should be binary, will be converted to -1 and 1).

        Returns
        -------
        self : object
            Returns self.
        """
        # Convert to numpy arrays
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        # Store number of features
        self.n_features_ = X.shape[1]

        # Convert labels to -1 and 1 if needed
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("PerceptronClassifier requires exactly 2 classes")

        # Map classes to -1 and 1
        self.class_mapping_ = {self.classes_[0]: -1, self.classes_[1]: 1}
        y_mapped = np.array([self.class_mapping_[val] for val in y])

        # Initialize weights and bias
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=self.n_features_)
        self.bias_ = np.float64(0.)
        self.errors_ = []

        # Training loop
        for _ in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X, y_mapped):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights_ += update * xi
                self.bias_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors == 0:
                break  # Converged

        return self

    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.weights_) + self.bias_

    def predict(self, X):
        """Return class label after unit step."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def predict_proba(self, X):
        """Return probability estimates for each class.
        
        Note: Perceptron doesn't naturally output probabilities, so we use
        a sigmoid transformation of the net input as an approximation.
        """
        # Simple approximation using sigmoid
        z = self.net_input(X)
        # Avoid overflow
        z = np.clip(z, -250, 250)
        prob_class_1 = 1.0 / (1.0 + np.exp(-z))
        prob_class_neg1 = 1.0 - prob_class_1
        return np.vstack([prob_class_neg1, prob_class_1]).T

    def score(self, X, y):
        """Return the accuracy score."""
        y_pred = self.predict(X)
        # Convert y_pred back to original class labels for comparison
        y_pred_original = np.array([list(self.class_mapping_.keys())[list(self.class_mapping_.values()).index(pred)] 
                                   for pred in y_pred])
        return np.mean(y_pred_original == y)