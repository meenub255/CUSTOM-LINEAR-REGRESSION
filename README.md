# Custom Linear Regression Library

A pure NumPy implementation of Linear Regression with support for multiple regularization techniques and solvers.

## Features

- **Multiple Penalty Types**: OLS (None), L1 (Lasso), L2 (Ridge)
- **Two Solvers**: Gradient Descent and Closed-form solution
- **Loss Functions**: Mean Squared Error (MSE) and Mean Absolute Error (MAE)
- **Bias/Intercept Support**: Properly handled in both solvers
- **Flexible Configuration**: Adjustable learning rate, iterations, and regularization strength

## Installation

Requirements: NumPy

```bash
pip install numpy
```

## Quick Start

```python
from linear_regression import LinearRegression
from generate_data import generate_linear_data

# Generate sample data
X_train, y_train = generate_linear_data(n_samples=100, n_features=2, noise=10.0)
X_test, y_test = generate_linear_data(n_samples=30, n_features=2, noise=10.0, seed=42)

# Create and train model
model = LinearRegression(penalty='l2', alpha=0.1, solver='closed')
model.fit(X_train, y_train)

# Evaluate
print(f"R^2 Score: {model.score(X_test, y_test):.4f}")
print(f"Predictions: {model.predict(X_test[:5])}")
```

## Usage

### LinearRegression API

```python
LinearRegression(
    penalty=None,      # 'l1', 'l2', or None (default: None)
    alpha=1.0,         # Regularization strength (default: 1.0)
    lr=0.01,           # Learning rate for GD (default: 0.01)
    n_iters=1000,      # Iterations for GD (default: 1000)
    solver='gd',       # 'gd' or 'closed' (default: 'gd')
    loss='mse'         # 'mse' or 'mae' (default: 'mse')
)
```

### Methods

- `fit(X, y)`: Fit the model on training data
- `predict(X)`: Predict target values for new data
- `score(X, y)`: Return R^2 coefficient of determination

## Examples

See `example_usage.py` for comprehensive examples including:
- OLS Regression
- Ridge Regression (L2)
- Lasso Regression (L1)
- Different loss functions (MSE, MAE)

## Solver Notes

- **Closed-form**: Fast, exact solution (OLS and Ridge only)
- **Gradient Descent**: Flexible, supports L1 and MAE loss
- Closed-form solution not available for L1 (Lasso) penalty
- Closed-form solver only supports MSE loss

## Model Features

- Intercept/bias term properly handled in both solvers
- Regularization applied only to weights, not bias
- Flexible loss functions for different use cases
