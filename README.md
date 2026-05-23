# Custom Linear Regression Library

A pure NumPy implementation of Linear Regression with support for multiple regularization techniques, solvers, and feature selection algorithms.

## Features

### Core Regression
- **Multiple Penalty Types**: OLS (None), L1 (Lasso), L2 (Ridge)
- **Two Solvers**: 
  - Gradient Descent (flexible, supports all penalties and loss functions)
  - Closed-form solution (fast, exact for OLS and Ridge)
- **Loss Functions**: Mean Squared Error (MSE) and Mean Absolute Error (MAE)
- **Bias/Intercept Support**: Properly handled in both solvers
- **Flexible Configuration**: Adjustable learning rate, iterations, and regularization strength

### Feature Selection
- **Forward Selection**: Greedily add features that most improve model performance
- **Backward Elimination**: Greedily remove features that least impact model performance
- **Cross-Validation Support**: Built-in k-fold cross-validation for robust feature evaluation

## Installation

Requirements: NumPy

```bash
pip install numpy
```

## Quick Start

### Basic Linear Regression

```python
from linear_regression import LinearRegression
from generate_data import generate_linear_data

# Generate sample data
X_train, y_train = generate_linear_data(n_samples=100, n_features=2, noise=10.0)
X_test, y_test = generate_linear_data(n_samples=30, n_features=2, noise=10.0, seed=42)

# OLS Regression
model = LinearRegression(penalty=None, solver='closed')
model.fit(X_train, y_train)

print(f"R^2 Score: {model.score(X_test, y_test):.4f}")
print(f"Weights: {model.weights}")
print(f"Intercept: {model.bias}")
```

### Ridge Regression

```python
# Ridge (L2) Regression
model = LinearRegression(penalty='l2', alpha=0.1, solver='closed')
model.fit(X_train, y_train)
print(f"R^2 Score: {model.score(X_test, y_test):.4f}")
```

### Lasso Regression

```python
# Lasso (L1) Regression with Gradient Descent
model = LinearRegression(penalty='l1', alpha=0.1, lr=0.01, n_iters=1000, 
                         solver='gd', loss='mse')
model.fit(X_train, y_train)
print(f"R^2 Score: {model.score(X_test, y_test):.4f}")
```

### Feature Selection

```python
from feature_selection import ForwardSelection, BackwardElimination

# Forward Selection
fwd = ForwardSelection(penalty=None, cv_splits=5)
fwd.fit(X_train, y_train, max_features=8)
selected = fwd.get_selected_features()
X_train_selected = fwd.transform(X_train)

# Backward Elimination
bwd = BackwardElimination(penalty=None, cv_splits=5)
bwd.fit(X_train, y_train, min_features=3)
selected = bwd.get_selected_features()
X_train_selected = bwd.transform(X_train)
```

## API Reference

### LinearRegression

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

**Methods:**
- `fit(X, y)`: Fit the model on training data
- `predict(X)`: Predict target values for new data
- `score(X, y)`: Return R² coefficient of determination

**Attributes:**
- `weights`: Model coefficients
- `bias`: Intercept/bias term

### ForwardSelection

```python
ForwardSelection(
    penalty=None,       # Regularization penalty
    alpha=1.0,          # Regularization strength
    cv_splits=5         # Number of CV folds
)
```

**Methods:**
- `fit(X, y, max_features=None)`: Perform forward selection
- `get_selected_features()`: Get indices of selected features
- `transform(X)`: Return only selected features

### BackwardElimination

```python
BackwardElimination(
    penalty=None,       # Regularization penalty
    alpha=1.0,          # Regularization strength
    cv_splits=5         # Number of CV folds
)
```

**Methods:**
- `fit(X, y, min_features=1)`: Perform backward elimination
- `get_selected_features()`: Get indices of selected features
- `transform(X)`: Return only selected features

## Examples

See `example_usage.py` for comprehensive examples including:
- OLS Regression
- Ridge Regression (L2)
- Lasso Regression (L1)
- Different loss functions (MSE, MAE)
- Forward Selection
- Backward Elimination

## Notes

### Solver Notes
- **Closed-form**: Fast, exact solution (OLS and Ridge only)
- **Gradient Descent**: Flexible, supports L1 and MAE loss
- Closed-form solution not available for L1 (Lasso) penalty
- Closed-form solver only supports MSE loss

### Feature Selection Notes
- Both algorithms use k-fold cross-validation for robust evaluation
- Forward Selection: Starts empty, adds best features iteratively
- Backward Elimination: Starts full, removes worst features iteratively
- Use `max_features` or `min_features` to control selection stopping criteria

### Model Features
- Intercept/bias term properly handled in both solvers
- Regularization applied only to weights, not bias
- Flexible loss functions for different use cases
- Cross-validation prevents overfitting during feature selection
