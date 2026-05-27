import numpy as np

def generate_linear_data(n_samples=100, n_features=2, noise=0.0, seed=None):
    """
    Generate synthetic linear regression data.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Number of features.
    noise : float
        Standard deviation of Gaussian noise added to the target.
    seed : int or None
        Random seed for reproducibility.
        
    Returns:
    --------
    X : array of shape (n_samples, n_features)
        Feature matrix.
    y : array of shape (n_samples,)
        Target values.
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    y = np.dot(X, true_weights) + np.random.normal(0, noise, n_samples)
    
    return X, y
