# Custom Linear Regression Library - Complete Documentation

This document provides comprehensive documentation for both Linear Regression and K-Nearest Neighbors (KNN) Regression implementations in the custom-linear-regression library.

## Table of Contents
1. [Overview](#overview)
2. [Linear Regression Documentation](#linear-regression-documentation)
3. [KNN Regression Documentation](#knn-regression-documentation)
4. [PCA Documentation](#pca-documentation)
5. [Decision Tree Documentation](#decision-tree-documentation)
6. [Perceptron Documentation](#perceptron-documentation)
7. [Installation](#installation)
8. [Usage Examples](#usage-examples)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The custom-linear-regression library is a pure NumPy-based implementation that provides:
- Linear Regression with regularization (L1/Lasso, L2/Ridge)
- K-Nearest Neighbors Regression with multiple distance metrics
- Feature selection methods (Forward Selection, Backward Elimination)
- Principal Component Analysis (PCA) for dimensionality reduction
- Decision Tree Regressor for non-linear regression
- Perceptron Classifier for binary classification
- Statistical diagnostics (Normality, Multicollinearity, Heteroscedasticity tests)
- Visualization tools (Matplotlib-based and text-based)
- Robust handling of missing values and outliers

All models follow a scikit-learn compatible API for easy integration.

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

## PCA Documentation

### Overview

The PCA class implements Principal Component Analysis for dimensionality reduction using Singular Value Decomposition (SVD). It provides:

- **Dimensionality Reduction**: Reduce the number of features while preserving maximum variance
- **Multiple Solvers**: Support for different SVD solvers (auto, full, arpack, randomized)
- **Whitening Option**: Transform components to have unit variance
- **Explained Variance Ratios**: Quantify how much variance each component captures
- **Reconstruction Capability**: Transform data back to original space
- **Scikit-Learn Compatible API**: Standard `fit()`, `transform()`, `fit_transform()` methods

### Key Features

1. **Flexible Component Selection**: Specify number of components as integer, float (proportion of variance), or None (all components)
2. **Multiple SVD Solvers**: Choose between 'auto', 'full', 'arpack', and 'randomized' solvers for efficiency
3. **Whitening**: Option to whiten components for uncorrelated outputs with unit variance
4. **Explained Variance Analysis**: Access to explained variance ratios and cumulative variance
5. **Data Reconstruction**: Ability to invert the transformation to approximate original data
6. **API Compatibility**: Follows scikit-learn's PCA API for easy integration

### Parameters

```
PCA(
    n_components=None,
    svd_solver='auto',
    whiten=False,
    random_state=None,
)
```

#### Parameter Details:

- `n_components` (int, float, None): 
  - If None: keeps all components (min(n_samples, n_features))
  - If int: specifies the number of components to keep
  - If float (0 < n_components < 1): specifies the proportion of variance to keep
- `svd_solver` (str): 
  - 'auto': automatic solver selection based on data shape
  - 'full': exact full SVD (uses LAPACK via scipy.svd)
  - 'arpack': truncated SVD using ARPACK (scipy.sparse.linalg.svds)
  - 'randomized': randomized SVD for large sparse matrices
- `whiten` (bool): When True, components are divided by sqrt(n_samples) * singular_values to ensure uncorrelated outputs with unit variance
- `random_state` (int, RandomState, None): Seed for reproducible results (used with randomized solver)

### Attributes

After fitting, the model has these attributes:

- `components_`: Principal axes in feature space (eigenvectors of covariance matrix)
- `explained_variance_`: Amount of variance explained by each component
- `explained_variance_ratio_`: Percentage of variance explained by each component
- `singular_values_`: Singular values corresponding to each component
- `mean_`: Per-feature mean used for centering
- `n_components_`: Estimated number of components
- `n_features_`: Number of features in training data

### Methods

- `fit(X, y=None)`: Fit the model with X (compute principal components)
- `transform(X)`: Apply dimensionality reduction to X
- `fit_transform(X, y=None)`: Fit to data, then transform it
- `inverse_transform(X)`: Transform data back to original space

### Usage Example

```python
import numpy as np
from custom_linear_regression import PCA

# Generate sample data with correlated features
np.random.seed(42)
X = np.random.randn(300, 3)
# Create correlation between features
X[:, 1] = X[:, 0] + 0.5*np.random.randn(300)
X[:, 2] = X[:, 0] + X[:, 1] + 0.5*np.random.randn(300)

# Create and fit PCA model
pca = PCA(n_components=2)  # Keep 2 components
pca.fit(X)

# Transform data
X_transformed = pca.transform(X)
print(f"Original shape: {X.shape}")
print(f"Transformed shape: {X_transformed.shape}")

# Check explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.2f}")

# Reconstruct data (approximation)
X_reconstructed = pca.inverse_transform(X_transformed)
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print(f"Reconstruction MSE: {reconstruction_error:.4f}")

# Get components (principal axes)
print(f"Components (principal axes):\n{pca.components_}")
```

### Integration with Regression Models

PCA can be used as a preprocessing step before regression to reduce dimensionality and handle multicollinearity:

```python
from custom_linear_regression import PCA, LinearRegression
from sklearn.model_selection import train_test_split

# Assume X, y are your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Fit regression on transformed data
model = LinearRegression()
model.fit(X_train_pca, y_train)

# Make predictions
y_pred = model.predict(X_test_pca)
```

### Notes

- The implementation centers the data but does not scale it. For scaled PCA, standardize the data before calling `fit`.
- The `svd_solver` parameter allows trade-offs between accuracy and speed:
  - 'full': Most accurate but slowest for large datasets
  - 'randomized': Fastest for large datasets, approximate
  - 'auto': Chooses between full and randomized based on data shape
- When `n_components` is a float, the solver defaults to 'full' to accurately compute explained variance
- Whitening is useful when you want uncorrelated features with unit variance (e.g., for some machine learning algorithms)

---

## Decision Tree Documentation

### Overview

The DecisionTreeRegressor class implements a decision tree for regression using a greedy, top-down approach. It provides:

- **Interpretable Models**: Produces human-readable decision rules
- **Non-linear Relationships**: Can capture complex non-linear patterns
- **Feature Importance**: Built-in feature importance calculation
- **Handling of Mixed Data**: Works with numerical features (can be extended for categorical)
- **Robust to Outliers**: Less sensitive to outliers than linear models
- **Scikit-Learn Compatible API**: Standard `fit()`, `predict()`, `score()` methods

### Key Features

1. **Recursive Binary Splitting**: Builds tree by recursively splitting data based on feature thresholds
2. **MSE Splitting Criterion**: Uses mean squared error to determine best splits
3. **Configurable Stopping Criteria**: Control tree complexity with max_depth, min_samples_split, min_samples_leaf
4. **Feature Importance**: Calculates importance based on total reduction in MSE
5. **Rule Extraction**: Can be converted to if-then rules for interpretation
6. **Handling of Missing Values**: Can be extended to handle missing data (currently requires clean input)

### Parameters

```
DecisionTreeRegressor(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
)
```

#### Parameter Details:

- `max_depth` (int or None): 
  - Maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain less than min_samples_split samples.
  - Controls overfitting: deeper trees can model more complex relationships but may overfit.
- `min_samples_split` (int): 
  - Minimum number of samples required to split an internal node.
  - Helps prevent overfitting by requiring sufficient data to make a split.
- `min_samples_leaf` (int): 
  - Minimum number of samples required to be at a leaf node.
  - Ensures leaves have sufficient data for reliable predictions.

### Attributes

After fitting, the model has these attributes:

- `tree_`: The decision tree structure (nested dictionary)
- `n_features_`: Number of features seen during fit
- `max_depth_`: The maximum depth of the fitted tree
- `n_leaves_`: Number of leaf nodes in the fitted tree

### Methods

- `fit(X, y)`: Build the decision tree from training data
- `predict(X)`: Predict regression targets for samples in X
- `score(X, y)`: Return the coefficient of determination R²
- `get_params()`: Get parameters of the estimator
- `set_params(**params)`: Set the parameters of the estimator

### Usage Example

```python
import numpy as np
from custom_linear_regression import DecisionTreeRegressor

# Generate sample data with non-linear relationship
np.random.seed(42)
X = np.random.randn(100, 2) * 10
# Create a non-linear target: y = x1^2 + x2 + noise
y = X[:, 0]**2 + X[:, 1] + np.random.randn(100) * 5

# Create and fit decision tree model
dt = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5)
dt.fit(X, y)

# Make predictions
y_pred = dt.predict(X[:5])
print(f"Predictions: {y_pred}")
print(f"R² Score: {dt.score(X, y):.4f}")

# Feature importance
print(f"Feature Importances: {dt.feature_importances_}")

# Tree depth and complexity
print(f"Tree Depth: {dt.max_depth_}")
print(f"Number of Leaves: {dt.n_leaves_}")
```

### Integration with Other Library Components

Decision trees can be combined with other library components for enhanced modeling:

```python
from custom_linear_regression import DecisionTreeRegressor, PCA, ForwardSelection
from sklearn.model_selection import train_test_split

# Assume X, y are your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Option 1: Use PCA for dimensionality reduction before decision tree
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

dt_pca = DecisionTreeRegressor(max_depth=4)
dt_pca.fit(X_train_pca, y_train)
print(f"Decision Tree + PCA R²: {dt_pca.score(X_test_pca, y_test):.4f}")

# Option 2: Use feature selection before decision tree
base_model = DecisionTreeRegressor(max_depth=4)
selector = ForwardSelection(base_model, k_features=5)
selector.fit(X_train, y_train)

X_train_selected = X_train[:, selector.k_feature_idx_]
X_test_selected = X_test[:, selector.k_feature_idx_]

dt_fs = DecisionTreeRegressor(max_depth=4)
dt_fs.fit(X_train_selected, y_train)
print(f"Decision Tree + Feature Selection R²: {dt_fs.score(X_test_selected, y_test):.4f}")

# Option 3: Use decision tree for feature importance to guide preprocessing
dt = DecisionTreeRegressor(max_depth=6)
dt.fit(X_train, y_train)
importances = dt.feature_importances_
print(f"Decision Tree Feature Importances: {importances}")

# Use importances to select top features
top_feature_indices = np.argsort(importances)[::-1][:3]  # Top 3 features
X_top = X[:, top_feature_indices]
X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(X_top, y, test_size=0.2, random_state=42)

dt_top = DecisionTreeRegressor(max_depth=4)
dt_top.fit(X_train_top, y_train_top)
print(f"Decision Tree (Top Features) R²: {dt_top.score(X_test_top, y_test_top):.4f}")
```

### Notes

- The implementation currently supports only numerical features. Categorical features need to be encoded numerically before use.
- The splitting criterion is mean squared error (MSE), which is appropriate for regression tasks.
- For classification tasks, a separate DecisionTreeClassifier would be needed (using Gini impurity or entropy).
- Decision trees are prone to overfitting; use max_depth, min_samples_split, and min_samples_leaf to control complexity.
- Ensemble methods like Random Forests or Gradient Boosting can be built upon this base decision tree implementation.
- The tree structure is stored as a nested dictionary, making it possible to extract and visualize decision rules.

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

### PCA Class

Located in `custom_linear_regression.pca.PCA`

### DecisionTreeRegressor Class

Located in `custom_linear_regression.decision_tree.DecisionTreeRegressor`

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