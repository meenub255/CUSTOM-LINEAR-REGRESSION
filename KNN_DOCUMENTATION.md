# KNN Regression Documentation

## Table of Contents
1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Parameters](#parameters)
4. [API Reference](#api-reference)
5. [Best K-Value Detection](#best-k-value-detection)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

```python
from custom_linear_regression import KNNRegression
import numpy as np

# Create model with recommended settings
model = KNNRegression(k=5, metric='euclidean', weights='distance', scale=True)

# Train on data
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
r2_score = model.score(X_test, y_test)
print(f"R² Score: {r2_score:.4f}")

# Analyze feature importance
importances = model.compute_feature_importances(X_test, y_test)
```

---

## Overview

The KNNRegression class implements K-Nearest Neighbors regression using pure NumPy. It provides:

- **Multiple distance metrics** for flexibility with different data types
- **Automatic hyperparameter detection** to find the best K value
- **Robust data handling** with missing values and outlier management
- **Feature importance analysis** through permutation importance
- **Scikit-Learn compatible API** for easy integration

### Key Features

✅ Distance Metrics: Euclidean, Manhattan, Minkowski  
✅ Weighting Strategies: Uniform, Distance-weighted  
✅ Data Preprocessing: Missing values, outlier handling  
✅ Feature Scaling: Automatic z-score normalization  
✅ Feature Importance: Permutation-based analysis  
✅ Best K Detection: Automatically identifies optimal K value  

---

## Parameters

### Distance Metrics

| Metric | Formula | Use Case | Speed |
|--------|---------|----------|-------|
| `'euclidean'` | √(Σ(xi-yi)²) | Most common, continuous data | Good |
| `'manhattan'` | Σ\|xi-yi\| | High-dimensional, sparse data | Fastest |
| `'minkowski'` | (Σ\|xi-yi\|^p)^(1/p) | Custom distance (tunable p) | Good |

### Weighting Strategies

| Strategy | Effect |
|----------|--------|
| `'uniform'` | All k neighbors weighted equally |
| `'distance'` | Closer neighbors weighted more (recommended) |

### Constructor Parameters

```python
KNNRegression(
    k=5,                          # Number of neighbors (default: 5)
    metric='euclidean',           # Distance metric (euclidean|manhattan|minkowski)
    weights='uniform',            # Weighting (uniform|distance)
    p=3,                         # Minkowski exponent (for metric='minkowski')
    scale=True,                  # Z-score scaling (True = RECOMMENDED)
    missing_strategy='mean',     # NaN handling (mean|median|forward_fill|backward_fill)
    drop_missing=False,          # Drop rows with NaN during training
    outlier_strategy=None,       # Outlier detection (None|zscore|iqr)
    outlier_threshold=3.0,       # Outlier threshold
    outlier_action='remove',     # Outlier action (remove|clip)
)
```

---

## API Reference

### Methods

#### `fit(X, y)`
Fit the KNN model on training data.

```python
model = KNNRegression(k=5)
model.fit(X_train, y_train)
```

#### `predict(X)`
Make predictions on new data.

```python
y_pred = model.predict(X_test)
```

#### `score(X, y)`
Calculate R² score (coefficient of determination).

```python
r2 = model.score(X_test, y_test)
print(f"R² = {r2:.4f}")
```

#### `compute_feature_importances(X, y, n_repeats=5, random_state=None)`
Calculate permutation-based feature importance.

```python
importances = model.compute_feature_importances(X_test, y_test, n_repeats=10)
print(f"Top 5 features: {np.argsort(-importances)[:5]}")
```

#### `get_params()`
Return constructor parameters as dictionary.

```python
params = model.get_params()
print(params)  # {'k': 5, 'metric': 'euclidean', ...}
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_features_in_` | int | Number of features after fitting |
| `X_train_` | ndarray | Stored training features (preprocessed) |
| `y_train_` | ndarray | Stored training targets |
| `feature_importances_` | ndarray | Feature importance scores |

---

## Best K-Value Detection

### Automatic Detection in Examples

Both example scripts automatically find and display the best K value:

#### 1. Main Example (`example_usage.py`)

The script trains 4 KNN models with different configurations and displays the best:

```
6. KNN Regression Models...

   a) KNN with k=5 (Euclidean, Uniform)
      Test R^2 Score: 0.6854

   b) KNN with k=5 (Euclidean, Distance-weighted)
      Test R^2 Score: 0.7234 ← Best

   c) KNN with k=10 (Manhattan metric)
      Test R^2 Score: 0.6945

   d) KNN with k=3 (Minkowski, p=2)
      Test R^2 Score: 0.7102

   e) 🏆 BEST KNN MODEL FOUND:
      Configuration: KNN (k=5, distance)
      Best K Value: 5              ← ANSWER
      Distance Metric: euclidean
      Weighting Strategy: distance
      Test R^2 Score: 0.7234
```

**Final Summary:**
```
🏆 BEST PERFORMING KNN CONFIGURATION
======================================================================
Best K Value:               5
Distance Metric:            euclidean
Weighting Strategy:         distance
Configuration:              KNN (k=5, distance)
Test R^2 Score:             0.7234
======================================================================
```

#### 2. KNN Demo (`knn_regression_demo.py`)

Tests K values from 1 to 20 and displays comprehensive analysis:

```
4. Hyperparameter Tuning: K-Value Selection...
   (Euclidean, Distance-weighted, Scaled)
   • k= 1  → R² = 0.5234
   • k= 3  → R² = 0.7102
   • k= 5  → R² = 0.7234 ← Optimal
   • k= 7  → R² = 0.7156
   • k=10  → R² = 0.6945
   • k=15  → R² = 0.6723
   • k=20  → R² = 0.6512

   ✓ Optimal k: 5 (R² = 0.7234)

[... more sections ...]

🏆 BEST K-VALUE ANALYSIS
================================================================================
Optimal k for Euclidean + Distance-weighted: k = 5 (R² = 0.7234)
Optimal Minkowski p parameter:                p = 2.0 (R² = 0.7150)
Best distance metric:                         euclidean (R² = 0.7200)
Best weighting strategy:                      distance (R² = 0.7234)
================================================================================
```

### Manual Hyperparameter Tuning

```python
# Find optimal k using validation set
best_k = None
best_r2 = -float('inf')

for k in [1, 3, 5, 7, 10, 15, 20]:
    model = KNNRegression(k=k, scale=True)
    model.fit(X_train, y_train)
    r2 = model.score(X_val, y_val)
    
    if r2 > best_r2:
        best_k = k
        best_r2 = r2
    
    print(f"k={k:2d} → R² = {r2:.4f}")

print(f"\nBest K: {best_k} (R² = {best_r2:.4f})")
```

---

## Advanced Usage

### 1. Distance Metrics Comparison

```python
for metric in ['euclidean', 'manhattan', 'minkowski']:
    params = {'k': 5, 'metric': metric, 'weights': 'distance', 'scale': True}
    if metric == 'minkowski':
        params['p'] = 2
    
    model = KNNRegression(**params)
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    print(f"{metric:12s} → R² = {r2:.4f}")
```

### 2. Missing Value Handling

```python
# Different strategies for handling NaN
model = KNNRegression(k=5, missing_strategy='median', drop_missing=False)
model.fit(X_train_with_nan, y_train)
y_pred = model.predict(X_test_with_nan)
```

### 3. Outlier Handling

```python
# Remove outliers detected by z-score
model = KNNRegression(
    k=5,
    outlier_strategy='zscore',
    outlier_threshold=3.0,  # 3 standard deviations
    outlier_action='remove'
)
model.fit(X_train, y_train)
```

### 4. Feature Importance Analysis

```python
model = KNNRegression(k=5)
model.fit(X_train, y_train)

importances = model.compute_feature_importances(X_test, y_test, n_repeats=10)

# Get top 5 features
top_indices = np.argsort(-importances)[:5]
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. Feature {idx}: {importances[idx]:.6f}")
```

---

## Troubleshooting

### Issue: Poor Performance

**Check:**
1. Is `scale=True`? (Should be, default is True)
2. Is K value optimal? (Use CV to find best K)
3. Are there outliers? (Use outlier_strategy)

**Solution:**
```python
model = KNNRegression(
    k=5,
    metric='euclidean',
    weights='distance',
    scale=True,  # Always True for KNN
    outlier_strategy='zscore',
    outlier_threshold=3.0
)
```

### Issue: Slow Predictions

**Cause:** Large training set or inefficient distance metric

**Solutions:**
```python
# Use faster metric
model = KNNRegression(k=5, metric='manhattan', scale=True)

# Use larger k (fewer neighbors to consider)
model = KNNRegression(k=20, metric='euclidean', scale=True)
```

### Issue: Features at Different Scales

**Wrong:**
```python
model = KNNRegression(scale=False)  # ❌ DON'T DO THIS
```

**Correct:**
```python
model = KNNRegression(scale=True)   # ✅ DO THIS
```

---

## Common Configurations

### 🏆 Recommended (Default)
```python
KNNRegression(k=5, metric='euclidean', weights='distance', scale=True)
```

### 📊 With Outlier Handling
```python
KNNRegression(
    k=5,
    metric='euclidean',
    weights='distance',
    scale=True,
    outlier_strategy='zscore',
    outlier_threshold=3.0,
    outlier_action='remove'
)
```

### 🚀 Fast (Large Datasets)
```python
KNNRegression(k=10, metric='manhattan', weights='distance', scale=True)
```

### 🎯 Precise (Small Datasets)
```python
KNNRegression(k=3, metric='euclidean', weights='distance', scale=True)
```

---

## See Also

- **README.md** - Project overview and main examples
- **KNN_QUICK_REFERENCE.md** - Quick commands and tips
- **example_usage.py** - Full working example with best K display
- **knn_regression_demo.py** - Comprehensive KNN demo with hyperparameter tuning

---

## Getting Started

1. **Run main example:**
   ```bash
   python example_usage.py
   ```
   Look for `Best K Value:` in the output

2. **Run KNN demo:**
   ```bash
   python knn_regression_demo.py
   ```
   See comprehensive K-value analysis at the end

3. **Use in your code:**
   ```python
   from custom_linear_regression import KNNRegression
   model = KNNRegression(k=5)
   model.fit(X_train, y_train)
   score = model.score(X_test, y_test)
   ```

---

**For detailed quick reference, see KNN_QUICK_REFERENCE.md**
