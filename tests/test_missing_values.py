"""
Test script demonstrating missing values handling in LinearRegression.
"""

import numpy as np
from linear_regression import LinearRegression

def create_data_with_missing_values():
    """Create sample data with missing values."""
    np.random.seed(42)
    X = np.random.randn(100, 4) * 10 + 50
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 0.8 * X[:, 3] + np.random.randn(100) * 5
    
    # Introduce missing values (NaN)
    missing_rate = 0.1
    for col in range(X.shape[1]):
        missing_idx = np.random.choice(X.shape[0], int(X.shape[0] * missing_rate), replace=False)
        X[missing_idx, col] = np.nan
    
    return X, y

def test_missing_value_strategies():
    """Test different missing value handling strategies."""
    X, y = create_data_with_missing_values()
    X_train = X[:80]
    y_train = y[:80]
    X_test = X[80:]
    y_test = y[80:]
    
    print("=" * 70)
    print("Testing Missing Values Handling Strategies")
    print("=" * 70)
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Missing values in training set: {np.isnan(X_train).sum()}")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Missing values in test set: {np.isnan(X_test).sum()}")
    
    strategies = ['mean', 'median', 'forward_fill', 'backward_fill']
    
    for strategy in strategies:
        print(f"\n{'-' * 70}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'-' * 70}")
        
        model = LinearRegression(
            penalty=None,
            solver='closed',
            missing_strategy=strategy,
            drop_missing=False
        )
        
        try:
            model.fit(X_train, y_train)
            r2_score = model.score(X_test, y_test)
            print(f"R² Score: {r2_score:.4f}")
            print(f"Weights: {model.weights}")
            print(f"Bias: {model.bias:.4f}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Test drop_missing strategy
    print(f"\n{'-' * 70}")
    print("Strategy: DROP MISSING (remove rows with NaN)")
    print(f"{'-' * 70}")
    
    model = LinearRegression(
        penalty=None,
        solver='closed',
        drop_missing=True
    )
    
    try:
        model.fit(X_train, y_train)
        r2_score = model.score(X_test, y_test)
        print(f"R² Score: {r2_score:.4f}")
        print(f"Weights: {model.weights}")
        print(f"Bias: {model.bias:.4f}")
    except Exception as e:
        print(f"Error: {e}")

def test_missing_values_with_regularization():
    """Test missing value handling with L1 and L2 regularization."""
    X, y = create_data_with_missing_values()
    X_train = X[:80]
    y_train = y[:80]
    X_test = X[80:]
    y_test = y[80:]
    
    print("\n" + "=" * 70)
    print("Testing Missing Values with Regularization")
    print("=" * 70)
    
    configs = [
        {'penalty': 'l2', 'alpha': 1.0, 'name': 'Ridge (L2) with Mean Imputation'},
        {'penalty': 'l1', 'alpha': 1.0, 'name': 'Lasso (L1) with Mean Imputation'},
    ]
    
    for config in configs:
        print(f"\n{'-' * 70}")
        print(f"{config['name']}")
        print(f"{'-' * 70}")
        
        if config['penalty'] == 'l1':
            model = LinearRegression(
                penalty=config['penalty'],
                alpha=config['alpha'],
                lr=0.01,
                n_iters=5000,
                solver='gd',
                loss='mse',
                missing_strategy='mean',
                drop_missing=False
            )
        else:
            model = LinearRegression(
                penalty=config['penalty'],
                alpha=config['alpha'],
                solver='closed',
                missing_strategy='mean',
                drop_missing=False
            )
        
        try:
            model.fit(X_train, y_train)
            r2_score = model.score(X_test, y_test)
            print(f"R² Score: {r2_score:.4f}")
            print(f"Weights: {model.weights}")
            print(f"Bias: {model.bias:.4f}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_missing_value_strategies()
    test_missing_values_with_regularization()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
