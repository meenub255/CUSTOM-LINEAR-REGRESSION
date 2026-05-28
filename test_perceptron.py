#!/usr/bin/env python3
"""
Test script for Perceptron Classifier implementation.
"""

import numpy as np
from custom_linear_regression import PerceptronClassifier

def test_perceptron_basic():
    """Test basic perceptron functionality."""
    print("Testing basic Perceptron...")
    
    # Generate sample binary classification data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    # Create a linearly separable target: y = 1 if x0 + x1 > 0 else -1
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    
    # Test with default parameters
    perceptron = PerceptronClassifier()
    perceptron.fit(X, y)
    
    # Check attributes
    assert hasattr(perceptron, 'weights_')
    assert hasattr(perceptron, 'bias_')
    assert hasattr(perceptron, 'errors_')
    assert perceptron.n_features_ == 2
    
    # Test predict
    y_pred = perceptron.predict(X)
    assert y_pred.shape == (100,)
    
    # Test score
    score = perceptron.score(X, y)
    assert 0 <= score <= 1
    
    print("+ Basic Perceptron test passed")

def test_perceptron_perfect_fit():
    """Test perceptron on perfectly separable data."""
    print("Testing Perceptron on perfect data...")
    
    # Create perfectly separable data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([-1, -1, -1, 1], dtype=float)  # AND-like: only [1,1] is positive
    
    # Create and fit perceptron model
    perceptron = PerceptronClassifier(learning_rate=0.1, n_epochs=100)
    perceptron.fit(X, y)
    
    y_pred = perceptron.predict(X)
    # Should be able to fit perfectly
    acc = perceptron.score(X, y)
    assert acc == 1.0  # Should achieve perfect accuracy
    
    print("+ Perfect fit Perceptron test passed")

def test_perceptron_parameters():
    """Test different parameter combinations."""
    print("Testing Perceptron parameters...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(50, 2)
    y = np.where(X[:, 0] > 0, 1, -1)
    
    # Test learning_rate effect
    perc_slow = PerceptronClassifier(learning_rate=0.001, n_epochs=1000)
    perc_slow.fit(X, y)
    
    perc_fast = PerceptronClassifier(learning_rate=0.1, n_epochs=1000)
    perc_fast.fit(X, y)
    
    # Both should converge, but fast might do it in fewer epochs
    # (not strictly guaranteed but likely)
    
    # Test n_epochs limitation
    perc_limited = PerceptronClassifier(learning_rate=0.01, n_epochs=5)
    perc_limited.fit(X, y)
    # Should have at most 5 epochs of errors recorded
    assert len(perc_limited.errors_) <= 5
    
    print("+ Perceptron parameters test passed")

def test_perceptron_not_fitted_error():
    """Test that NotFittedError is raised when appropriate."""
    print("Testing NotFittedError...")
    
    perceptron = PerceptronClassifier()
    
    try:
        perceptron.predict(np.random.randn(10, 2))
        assert False, "Should have raised NotFittedError"
    except Exception as e:
        assert "not fitted" in str(e).lower()
    
    try:
        perceptron.score(np.random.randn(10, 2), np.random.randint(0, 2, 10))
        assert False, "Should have raised NotFittedError"
    except Exception as e:
        assert "not fitted" in str(e).lower()
    
    print("+ NotFittedError test passed")

def test_perceptron_predict_proba():
    """Test that predict_proba returns valid probabilities."""
    print("Testing Perceptron predict_proba...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(20, 2)
    y = np.where(X[:, 0] > 0, 1, -1)
    
    perceptron = PerceptronClassifier()
    perceptron.fit(X, y)
    
    probs = perceptron.predict_proba(X)
    
    # Check shape
    assert probs.shape == (20, 2)
    
    # Check that probabilities sum to 1 (approximately)
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0)
    
    # Check that probabilities are between 0 and 1
    assert np.all(probs >= 0) and np.all(probs <= 1)
    
    print("+ Perceptron predict_proba test passed")

def test_perceptron_different_labels():
    """Test perceptron with different binary label formats."""
    print("Testing Perceptron with different label formats...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(50, 2)
    
    # Test with 0/1 labels
    y_01 = np.where(X[:, 0] > 0, 1, 0)
    perceptron_01 = PerceptronClassifier()
    perceptron_01.fit(X, y_01)
    acc_01 = perceptron_01.score(X, y_01)
    
    # Test with string labels
    y_str = np.where(X[:, 0] > 0, 'positive', 'negative')
    perceptron_str = PerceptronClassifier()
    perceptron_str.fit(X, y_str)
    acc_str = perceptron_str.score(X, y_str)
    
    # Both should work
    assert acc_01 >= 0.0 and acc_01 <= 1.0
    assert acc_str >= 0.0 and acc_str <= 1.0
    
    print("+ Perceptron different labels test passed")

def main():
    """Run all tests."""
    print("Testing Perceptron Classifier implementation\n")
    
    test_perceptron_basic()
    test_perceptron_perfect_fit()
    test_perceptron_parameters()
    test_perceptron_not_fitted_error()
    test_perceptron_predict_proba()
    test_perceptron_different_labels()
    
    print("\n* All Perceptron tests passed!")

if __name__ == "__main__":
    main()