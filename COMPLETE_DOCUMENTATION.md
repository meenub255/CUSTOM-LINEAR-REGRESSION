# Custom Linear Regression Library - Complete Documentation

This document provides comprehensive documentation for both Linear Regression and K-Nearest Neighbors (KNN) Regression implementations in the custom-linear-regression library.

## Table of Contents
1. [Overview](#overview)
2. [Linear Regression Documentation](#linear-regression-documentation)
3. [KNN Regression Documentation](#knn-regression-documentation)
4. [Installation](#installation)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The custom-linear-regression library is a pure NumPy-based implementation that provides:
- Linear Regression with regularization (L1/Lasso, L2/Ridge)
- K-Nearest Neighbors Regression with multiple distance metrics
- Feature selection methods (Forward Selection, Backward Elimination)
- Statistical diagnostics (Normality, Multicollinearity, Heteroscedasticity tests)
- Visualization tools (Matplotlib-based and text-based)
- Robust handling of missing values and outliers

Both regression models follow a scikit-learn compatible API for easy integration.

---

## Linear Regression Documentation

### Features

The LinearRegression class implements ordinary least squares regression with extensions:

1. **ADD INTERCEPT**: Fully integrated intercept handling with `fit_intercept=True` and Scikit-learn style `.coef_` and `.intercept_` properties.
2. **FORWARD SELECTION**: Stepwise greedy feature addition based on model evaluation score.
3. **BACKWARD ELIMINATION**: Stepwise greedy feature removal to find the optimal subset.
4. **L1 REGULARIZATION**: Lasso regularization via Gradient Descent (`penalty='l1'`).
5. **L2 REGULARIZATION**: Ridge regularization via Gradient Descent or Closed-form solver (`penalty='l2'`).
6. **MISSING VALUES**: Robust imputation strategies (`mean`, `median`, `forward_fill`, `backward_fill`) and `drop_missing`.
7. **OUTLIERS**: Detection and handling using Z-score and IQR strategies.
8. **NORMALITY**: Shapiro-Wilk, Anderson-Darling, Jarque-Bera, and Kolmogorov-Smirnov tests.
9. **MULTICOLLINEARITY**: Variance Inflation Factor (VIF) scores and Correlation matrices.
10. **HETEROSCEDASTICITY**: Breusch-Pagan, White, and Goldfeld-Quandt tests.
11. **VISUALIZATION**: Matplotlib-based diagnostic plots and pure-text ASCII/ANSI visualizers for environments without graphical libraries.
12. **PREDICTION**: Fast, vectorized predictions maintaining handling for missing and outlier data.

### Parameters

```
LinearRegression(
    fit_intercept=True,
    penalty=None,
    alpha=1.0,
    lr=0.01,
    n_iters=1000,
    solver="gd",
    loss="mse",
    missing_strategy="mean",
    drop_missing=False,
    outlier_strategy=None,
    outlier_threshold=3.0,
    outlier_action="remove",
)
```

#### Parameter Details:

- `fit_intercept` (bool): Whether to calculate the intercept for this model.
- `penalty` (str or None): Regularization penalty. Can be None (OLS), 'l1' (Lasso), or 'l2' (Ridge).
- `alpha` (float): Regularization strength.
- `lr` (float): Learning rate for Gradient Descent solver.
- `n_iters` (int): Number of iterations for Gradient Descent solver.
- `solver` (str): 'gd' (Gradient Descent) or 'closed' (Closed-form solution). Note: 'closed' is only available for None and 'l2' penalty.
- `loss` (str): Loss function to optimize. Can be 'mse' (Mean Squared Error) or 'mae' (Mean Absolute Error). Note: 'closed' solver only supports 'mse'.
- `missing_strategy` (str): Strategy for handling missing values. Options: 'mean', 'median', 'forward_fill', 'backward_fill'. Default: 'mean'.
- `drop_missing` (bool): Whether to drop rows with missing values instead of imputing.
- `outlier_strategy` (str or None): Strategy for detecting outliers. Options: None, 'zscore', 'iqr'.
- `outlier_threshold` (float): Threshold for outlier detection (z-score or IQR multiplier).
- `outlier_action` (str): Action to take for outliers: 'remove', 'cap', or None.

### Attributes

After fitting, the model has these attributes:
- `coef_`: Coefficients of the features
- `intercept_`: Intercept term
- `feature_importances_`: Permutation-based feature importance
- `n_features_in_`: Number of features seen during fit
- `feature_names_in_`: Names of features seen during fit

### Methods

- `fit(X, y)`: Fit the model to training data
- `predict(X)`: Predict using the linear model
- `score(X, y)`: Return the coefficient of determination R²
- `compute_feature_importances(X, y)`: Compute permutation-based feature importances

---

## KNN Regression Documentation

### Overview

The KNNRegression class implements K-Nearest Neighbors regression using pure NumPy. It provides:

- **Multiple distance metrics** for flexibility with different data types
- **Automatic hyperparameter detection** to find the best K value
- **Robust data handling** with missing values and outlier management
- **Feature importance analysis** through permutation importance
- **Scikit-Learn compatible API** for easy integration

### Key Features

1. **Multiple Distance Metrics**: Euclidean, Manhattan, Minkowski (fully customizable)
2. **Weighting Strategies**: Uniform averaging or inverse-distance weighting
3. **Data Preprocessing**: Automatic handling of missing values and outliers
4. **Feature Scaling**: Z-score normalization for distance computation
5. **Feature Importance**: Permutation-based importance analysis
6. **Hyperparameter Tuning**: Support for k, metric, and weights optimization
7. **Best K-Value Detection**: Automatically identifies and displays optimal K value with R² score
8. **Scikit-Learn Compatible**: Standard `fit()`, `predict()`, `score()` API

### Parameters

```
KNNRegression(
    k=5,
    metric='euclidean',
    weights='uniform',
    p=2,
    scale=True,
    missing_strategy='mean',
    drop_missing=False,
    outlier_strategy=None,
    outlier_threshold=3.0,
    outlier_action='remove',
)
```

#### Parameter Details:

- `k` (int): Number of neighbours to use for prediction (default: 5)
- `metric` (str): Distance metric to use. Options: 'euclidean', 'manhattan', 'minkowski'
- `weights` (str): Weight function used in prediction. Options: 'uniform', 'distance'
- `p` (int): Minkowski exponent (only used when metric='minkowski')
- `scale` (bool): Whether to apply z-score feature scaling before distance computation (default: True)
- `missing_strategy` (str): Strategy for handling missing values. Options: 'mean', 'median', 'forward_fill', 'backward_fill'. Default: 'mean'.
- `drop_missing` (bool): Whether to drop rows with missing values instead of imputing.
- `outlier_strategy` (str or None): Strategy for detecting outliers. Options: None, 'zscore', 'iqr'.
- `outlier_threshold` (float): Threshold for outlier detection (z-score or IQR multiplier).
- `outlier_action` (str): Action to take for outliers: 'remove', 'cap', or None.

### Attributes

After fitting, the model has these attributes:
- `X_train_`: Training data (after preprocessing)
- `y_train_`: Target values (after preprocessing)
- `feature_importances_`: Permutation-based feature importance
- `n_features_in_`: Number of features seen during fit
- `feature_names_in_`: Names of features seen during fit
- `best_k_`: Optimal k value found (if auto-tuning was used)
- `best_score_`: R² score of the best k value

### Methods

- `fit(X, y)`: Fit the model using training data
- `predict(X)`: Predict the target for the provided data
- `score(X, y)`: Return the coefficient of determination R²
- `compute_feature_importances(X, y)`: Compute permutation-based feature importances
- `find_best_k(X, y, k_range=None)`: Find optimal k value using cross-validation

---

## Installation

Install the package locally:
```bash
python -m pip install .
```

For editable development install:
```bash
python -m pip install -e .
```

### Dependencies

- numpy>=1.24
- scipy>=1.10

### Optional Dependencies

- viz: matplotlib>=3.7 (for visualization)
- scrape: playwright>=1.54 (for web scraping utilities)

---

## Usage Examples

### Linear Regression Example

```python
import numpy as np
from custom_linear_regression import LinearRegression

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2 + 1.5*X[:,0] - 2.0*X[:,1] + 0.5*X[:,2] + np.random.randn(100)*0.1

# Create and fit model
model = LinearRegression(fit_intercept=True, penalty=None)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X[:5])
print("Predictions:", y_pred)
print("R² Score:", model.score(X, y))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```

### KNN Regression Example

```python
import numpy as np
from custom_linear_regression import KNNRegression

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.sum(X**2, axis=1) + np.random.randn(100)*0.1  # Non-linear relationship

# Create and fit model
model = KNNRegression(k=5, metric='euclidean', weights='distance', scale=True)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X[:5])
print("Predictions:", y_pred)
print("R² Score:", model.score(X, y))

# Find best k value
best_k = model.find_best_k(X, y, k_range=range(1, 21))
print("Best k value:", best_k)

# Feature importance
importances = model.compute_feature_importances(X, y)
print("Feature importances:", importances)
```

### Advanced Usage with Feature Selection

```python
from custom_linear_regression import LinearRegression, ForwardSelection

# Create base model
base_model = LinearRegression()

# Apply forward selection
selector = ForwardSelection(base_model, k_features=5)
selector.fit(X, y)

# Get selected features
selected_features = selector.k_feature_idx_
print("Selected features:", selected_features)

# Fit final model with selected features
final_model = LinearRegression()
final_model.fit(X[:, selected_features], y)
```

---

## API Reference

### LinearRegression Class

Located in `custom_linear_regression.linear_regression.LinearRegression`

### KNNRegression Class

Located in `custom_linear_regression.knn_regression.KNNRegression`

### Feature Selection Classes

- `ForwardSelection`: Located in `custom_linear_regression.feature_selection.ForwardSelection`
- `BackwardElimination`: Located in `custom_linear_regression.feature_selection.BackwardElimination`

### Diagnostics Classes

- `NormalityTest`: Located in `custom_linear_regression.diagnostics.NormalityTest`
- `MulticollinearityTest`: Located in `custom_linear_regression.diagnostics.MulticollinearityTest`
- `HeteroscedasticityTest`: Located in `custom_linear_regression.diagnostics.HeteroscedasticityTest`
- `OutlierDetector`: Located in `custom_linear_regression.diagnostics.OutlierDetector`

### Visualization Classes

- `RegressionVisualizer`: Located in `custom_linear_regression.visualization.RegressionVisualizer`
- `TextVisualizer`: Located in `custom_linear_regression.visualization.TextVisualizer`

### Exception Classes

All exceptions are located in `custom_linear_regression.exceptions`:
- `CustomLinearRegressionError`: Base exception class
- `NotFittedError`: Raised when model is used before fitting
- `OptimizationFailedError`: Raised when optimization fails to converge
- `DataQualityError`: Raised when data quality issues are detected

---

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you have installed the package correctly with `pip install .`
2. **Missing Dependencies**: Install required packages with `pip install numpy scipy`
3. **Visualization Issues**: Install optional dependencies with `pip install -e .[viz]`
4. **Memory Errors**: For large datasets, consider reducing complexity or using sparse representations
5. **Convergence Issues**: For gradient descent solvers, try adjusting learning rate (`lr`) or number of iterations (`n_iters`)

### Getting Help

If you encounter issues not covered here, please check:
1. The GitHub repository issues page
2. The examples in the `example_usage.py` and `knn_regression_demo.py` files
3. The unit tests in the `test_*.py` files

### Version Information

To check the installed version:
```python
import custom_linear_regression
print(custom_linear_regression.__version__)
```

Current version: 0.2.0

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.