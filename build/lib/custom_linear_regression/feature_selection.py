import numpy as np
from .linear_regression import LinearRegression


class ForwardSelection:
    """
    Forward Selection: Start with no features and iteratively add the most significant feature.
    
    This is a greedy feature selection algorithm that:
    1. Starts with an empty set of features
    2. At each step, adds the feature that improves the model performance the most
    3. Continues until a stopping criterion is met (max features or no improvement)
    """
    
    def __init__(self, penalty=None, alpha=1.0, cv_splits=5, solver=None, loss='mse', lr=0.01, n_iters=1000):
        """
        Parameters:
        -----------
        penalty : str or None
            Regularization penalty for the underlying LinearRegression model.
        alpha : float
            Regularization strength.
        cv_splits : int
            Number of cross-validation splits for model evaluation.
        """
        self.penalty = penalty
        self.alpha = alpha
        self.cv_splits = cv_splits
        self.solver = solver
        self.loss = loss
        self.lr = lr
        self.n_iters = n_iters
        self.selected_features = []
        self.feature_scores = []
        self.best_score = -np.inf
        
    def fit(self, X, y, max_features=None):
        """
        Fit the forward selection algorithm.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        max_features : int or None
            Maximum number of features to select. If None, select all features.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        n_samples, n_features = X.shape
        if max_features is None:
            max_features = n_features
            
        all_features = set(range(n_features))
        self.selected_features = []
        self.feature_scores = []
        self.best_score = -np.inf
        
        for _ in range(min(max_features, n_features)):
            remaining_features = all_features - set(self.selected_features)
            best_feature = None
            best_score = self.best_score
            
            for feature in remaining_features:
                # Evaluate model with this feature added
                candidate_features = self.selected_features + [feature]
                X_subset = X[:, candidate_features]
                
                # Simple train-test evaluation
                score = self._evaluate(X_subset, y)
                
                if score > best_score:
                    best_score = score
                    best_feature = feature
            
            # If no improvement, stop
            if best_feature is None:
                break
                
            self.selected_features.append(best_feature)
            self.feature_scores.append(best_score)
            self.best_score = best_score
    
    def _evaluate(self, X, y):
        """Evaluate model performance using cross-validation."""
        n_samples = X.shape[0]
        fold_size = n_samples // self.cv_splits
        scores = []
        
        for fold in range(self.cv_splits):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.cv_splits - 1 else n_samples
            
            X_train = np.vstack([X[:test_start], X[test_end:]])
            y_train = np.concatenate([y[:test_start], y[test_end:]])
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            solver = self.solver
            if solver is None:
                solver = 'gd' if self.penalty == 'l1' else 'closed'

            model = LinearRegression(
                penalty=self.penalty,
                alpha=self.alpha,
                solver=solver,
                loss=self.loss,
                lr=self.lr,
                n_iters=self.n_iters,
            )
            try:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                scores.append(score)
            except:
                scores.append(-np.inf)
        
        return np.mean(scores)
    
    def get_selected_features(self):
        """Return indices of selected features."""
        return self.selected_features
    
    def transform(self, X):
        """Return only the selected features."""
        X = np.array(X, dtype=float)
        return X[:, self.selected_features]


class BackwardElimination:
    """
    Backward Elimination: Start with all features and iteratively remove the least significant feature.
    
    This is a greedy feature selection algorithm that:
    1. Starts with all features
    2. At each step, removes the feature that has the least impact on model performance
    3. Continues until a stopping criterion is met (min features or no improvement)
    """
    
    def __init__(self, penalty=None, alpha=1.0, cv_splits=5, solver=None, loss='mse', lr=0.01, n_iters=1000):
        """
        Parameters:
        -----------
        penalty : str or None
            Regularization penalty for the underlying LinearRegression model.
        alpha : float
            Regularization strength.
        cv_splits : int
            Number of cross-validation splits for model evaluation.
        """
        self.penalty = penalty
        self.alpha = alpha
        self.cv_splits = cv_splits
        self.solver = solver
        self.loss = loss
        self.lr = lr
        self.n_iters = n_iters
        self.selected_features = []
        self.feature_scores = []
        self.best_score = -np.inf
        
    def fit(self, X, y, min_features=1):
        """
        Fit the backward elimination algorithm.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        min_features : int
            Minimum number of features to retain.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        n_samples, n_features = X.shape
        self.selected_features = list(range(n_features))
        
        # Evaluate initial model with all features
        X_subset = X[:, self.selected_features]
        self.best_score = self._evaluate(X_subset, y)
        self.feature_scores = [self.best_score]
        
        while len(self.selected_features) > min_features:
            worst_feature = None
            worst_drop_score = self.best_score
            
            for feature in self.selected_features:
                # Evaluate model with this feature removed
                candidate_features = [f for f in self.selected_features if f != feature]
                if not candidate_features:
                    continue
                    
                X_subset = X[:, candidate_features]
                score = self._evaluate(X_subset, y)
                
                if score > worst_drop_score:
                    worst_drop_score = score
                    worst_feature = feature
            
            # If removing a feature improves or maintains score, remove it
            if worst_feature is not None and worst_drop_score >= self.best_score:
                self.selected_features.remove(worst_feature)
                self.best_score = worst_drop_score
                self.feature_scores.append(self.best_score)
            else:
                # No improvement, stop
                break
    
    def _evaluate(self, X, y):
        """Evaluate model performance using cross-validation."""
        n_samples = X.shape[0]
        fold_size = n_samples // self.cv_splits
        scores = []
        
        for fold in range(self.cv_splits):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.cv_splits - 1 else n_samples
            
            X_train = np.vstack([X[:test_start], X[test_end:]])
            y_train = np.concatenate([y[:test_start], y[test_end:]])
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            solver = self.solver
            if solver is None:
                solver = 'gd' if self.penalty == 'l1' else 'closed'

            model = LinearRegression(
                penalty=self.penalty,
                alpha=self.alpha,
                solver=solver,
                loss=self.loss,
                lr=self.lr,
                n_iters=self.n_iters,
            )
            try:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                scores.append(score)
            except:
                scores.append(-np.inf)
        
        return np.mean(scores)
    
    def get_selected_features(self):
        """Return indices of selected features."""
        return self.selected_features
    
    def transform(self, X):
        """Return only the selected features."""
        X = np.array(X, dtype=float)
        return X[:, self.selected_features]