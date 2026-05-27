#!/usr/bin/env python3
"""
Test script for PCA implementation.
"""

import numpy as np
from custom_linear_regression import PCA

def test_pca_basic():
    """Test basic PCA functionality."""
    print("Testing basic PCA...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    # Create some correlation
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(100)
    X[:, 2] = X[:, 0] + X[:, 1] + 0.5 * np.random.randn(100)
    
    # Test with n_components as int
    pca = PCA(n_components=2)
    pca.fit(X)
    
    # Check attributes
    assert hasattr(pca, 'components_')
    assert hasattr(pca, 'explained_variance_')
    assert hasattr(pca, 'explained_variance_ratio_')
    assert hasattr(pca, 'mean_')
    assert pca.n_components_ == 2
    assert pca.components_.shape == (2, 5)
    
    # Test transform
    X_transformed = pca.transform(X)
    assert X_transformed.shape == (100, 2)
    
    # Test fit_transform
    X_fit_transform = pca.fit_transform(X)
    np.testing.assert_array_almost_equal(X_transformed, X_fit_transform)
    
    # Test inverse transform
    X_reconstructed = pca.inverse_transform(X_transformed)
    assert X_reconstructed.shape == (100, 5)
    
    # Check that explained variance ratio sums to <= 1
    assert np.sum(pca.explained_variance_ratio_) <= 1.0
    
    print("+ Basic PCA test passed")

def test_pca_variance_threshold():
    """Test PCA with variance threshold."""
    print("Testing PCA with variance threshold...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    # Test with n_components as float (variance proportion)
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    pca.fit(X)
    
    # Should keep enough components to explain 95% variance
    assert np.sum(pca.explained_variance_ratio_) >= 0.95
    assert pca.n_components_ >= 1
    
    print("+ Variance threshold PCA test passed")

def test_pca_whiten():
    """Test PCA with whitening."""
    print("Testing PCA with whitening...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    # Test with whitening
    pca = PCA(n_components=3, whiten=True)
    pca.fit(X)
    
    X_transformed = pca.transform(X)
    
    # With whitening, transformed data should have unit variance
    # (approximately, due to sampling)
    var_of_transformed = np.var(X_transformed, axis=0)
    np.testing.assert_array_almost_equal(var_of_transformed, np.ones(3), decimal=1)
    
    print("+ Whitening PCA test passed")

def test_pca_not_fitted_error():
    """Test that NotFittedError is raised when appropriate."""
    print("Testing NotFittedError...")
    
    pca = PCA(n_components=2)
    
    try:
        pca.transform(np.random.randn(10, 5))
        assert False, "Should have raised NotFittedError"
    except Exception as e:
        assert "not fitted" in str(e).lower()
    
    try:
        pca.inverse_transform(np.random.randn(10, 2))
        assert False, "Should have raised NotFittedError"
    except Exception as e:
        assert "not fitted" in str(e).lower()
    
    print("+ NotFittedError test passed")

def main():
    """Run all tests."""
    print("Testing PCA implementation\n")
    
    test_pca_basic()
    test_pca_variance_threshold()
    test_pca_whiten()
    test_pca_not_fitted_error()
    
    print("\n*** All PCA tests passed!")

if __name__ == "__main__":
    main()