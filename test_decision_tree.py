#!/usr/bin/env python3
"""
Test script for Decision Tree Regressor implementation.
"""

import numpy as np
from custom_linear_regression import DecisionTreeRegressor

def test_decision_tree_basic():
    """Test basic decision tree functionality."""
    print("Testing basic Decision Tree...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X[:, 0] * 2 + X[:, 1] * -1.5 + X[:, 2] * 0.5 + np.random.randn(100) * 0.1
    
    # Test with default parameters
    dt = DecisionTreeRegressor()
    dt.fit(X, y)
    
    # Check attributes
    assert hasattr(dt, 'tree_')
    assert hasattr(dt, 'n_features_')
    assert dt.n_features_ == 3
    
    # Test predict
    y_pred = dt.predict(X)
    assert y_pred.shape == (100,)
    
    # Test score
    score = dt.score(X, y)
    assert 0 <= score <= 1
    
    print("+ Basic Decision Tree test passed")

def test_decision_tree_perfect_fit():
    """Test decision tree on perfectly separable data."""
    print("Testing Decision Tree on perfect data...")
    
    # Create perfectly separable data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)  # XOR-like
    
    # Use a deep enough tree to capture the pattern
    dt = DecisionTreeRegressor(max_depth=3, min_samples_split=2, min_samples_leaf=1)
    dt.fit(X, y)
    
    y_pred = dt.predict(X)
    # Should be able to fit perfectly (or very close)
    mse = np.mean((y - y_pred) ** 2)
    assert mse < 0.1  # Allow small numerical errors
    
    print("+ Perfect fit Decision Tree test passed")

def test_decision_tree_parameters():
    """Test different parameter combinations."""
    print("Testing Decision Tree parameters...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = X[:, 0] ** 2 + np.random.randn(50) * 0.1
    
    # Test max_depth constraint
    dt_shallow = DecisionTreeRegressor(max_depth=2)
    dt_shallow.fit(X, y)
    assert dt_shallow.max_depth_ <= 2
    
    dt_deep = DecisionTreeRegressor(max_depth=10)
    dt_deep.fit(X, y)
    # Should be able to grow deeper (though may hit other limits)
    
    # Test min_samples_split
    dt_large_split = DecisionTreeRegressor(min_samples_split=50)  # Can't split
    dt_large_split.fit(X, y)
    # Should create a single leaf (no splits possible)
    
    # Test min_samples_leaf
    dt_large_leaf = DecisionTreeRegressor(min_samples_leaf=30)  # Need 30 in each leaf
    dt_large_leaf.fit(X, y)
    # Should be constrained
    
    print("+ Decision Tree parameters test passed")

def test_decision_tree_not_fitted_error():
    """Test that NotFittedError is raised when appropriate."""
    print("Testing NotFittedError...")
    
    dt = DecisionTreeRegressor(max_depth=3)
    
    try:
        dt.predict(np.random.randn(10, 3))
        assert False, "Should have raised NotFittedError"
    except Exception as e:
        assert "not fitted" in str(e).lower()
    
    try:
        dt.score(np.random.randn(10, 3), np.random.randn(10))
        assert False, "Should have raised NotFittedError"
    except Exception as e:
        assert "not fitted" in str(e).lower()
    
    print("+ NotFittedError test passed")

def test_decision_tree_feature_importance():
    """Test that feature importance is computed correctly."""
    print("Testing Decision Tree feature importance...")
    
    # Create data where first feature is more important
    np.random.seed(42)
    X = np.random.randn(100, 3)
    # Make first feature strongly predictive
    y = X[:, 0] * 3 + X[:, 1] * 0.5 + X[:, 2] * 0.1 + np.random.randn(100) * 0.1
    
    dt = DecisionTreeRegressor(max_depth=5)
    dt.fit(X, y)
    
    importances = dt.feature_importances_
    assert len(importances) == 3
    assert np.all(importances >= 0)  # Importances should be non-negative
    assert np.isclose(np.sum(importances), 1.0)  # Should sum to 1 (approximately)
    
    # First feature should have highest importance
    assert importances[0] >= importances[1]
    assert importances[0] >= importances[2]
    
    print("+ Decision Tree feature importance test passed")

def main():
    """Run all tests."""
    print("Testing Decision Tree Regressor implementation\n")
    
    test_decision_tree_basic()
    test_decision_tree_perfect_fit()
    test_decision_tree_parameters()
    test_decision_tree_not_fitted_error()
    test_decision_tree_feature_importance()
    
    print("\n* All Decision Tree tests passed!")

if __name__ == "__main__":
    main()