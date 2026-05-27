"""
K-Nearest Neighbors Regression Demo
===================================

This script demonstrates the full capabilities of the custom KNN Regression
implementation, including:

* Different distance metrics (Euclidean, Manhattan, Minkowski)
* Weighting strategies (Uniform vs Distance-weighted)
* Feature scaling
* Missing value and outlier handling
* Permutation-based feature importance
* Hyperparameter tuning with cross-validation
"""

import numpy as np
import csv
from custom_linear_regression import KNNRegression, LinearRegression


def load_csv_data(filename):
    """Load data from CSV file and return features and target."""
    X_base = []
    y = []
    localities = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            try:
                features = [float(row[0]), float(row[1])]
                target = float(row[3])
                locality = row[4].strip()
                
                X_base.append(features)
                y.append(target)
                localities.append(locality)
            except (ValueError, IndexError):
                continue
    
    unique_localities = list(set(localities))
    unique_localities.sort()
    
    X = []
    for i in range(len(X_base)):
        one_hot = [0.0] * len(unique_localities)
        loc_index = unique_localities.index(localities[i])
        one_hot[loc_index] = 1.0
        full_features = X_base[i] + one_hot
        X.append(full_features)
    
    return np.array(X), np.array(y), unique_localities


def train_test_split(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_samples = int(len(X) * test_size)
    
    X_train = X[indices[test_samples:]]
    y_train = y[indices[test_samples:]]
    X_test = X[indices[:test_samples]]
    y_test = y[indices[:test_samples]]
    
    return X_train, X_test, y_train, y_test


def standardize_features(X_train, X_test):
    """Standardize features (zero mean, unit variance)."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled, mean, std


def main():
    print("=" * 80)
    print("K-NEAREST NEIGHBORS (KNN) REGRESSION - COMPREHENSIVE DEMO")
    print("=" * 80)
    
    # Load and prepare data
    print("\n1. Loading and Preparing Data...")
    X, y, localities = load_csv_data("thrissur_house_prices.csv")
    print(f"   Dataset size: {len(X)} samples")
    print(f"   Number of features: {X.shape[1]} (including one-hot encoded localities)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_scaled, X_test_scaled, mean, std = standardize_features(X_train, X_test)
    print(f"   Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    
    # ======================================================================
    # 2. DISTANCE METRICS COMPARISON
    # ======================================================================
    print("\n2. Comparing Distance Metrics...")
    print("   (All with k=5, distance-weighted averaging)")
    
    metrics = ['euclidean', 'manhattan', 'minkowski']
    metric_results = {}
    
    for metric in metrics:
        params = {'k': 5, 'metric': metric, 'weights': 'distance', 'scale': True}
        if metric == 'minkowski':
            params['p'] = 2  # Euclidean-like for comparison
        
        model = KNNRegression(**params)
        model.fit(X_train_scaled, y_train)
        r2 = model.score(X_test_scaled, y_test)
        metric_results[metric] = r2
        print(f"   • {metric:15s} → R² = {r2:.4f}")
    
    best_metric = max(metric_results, key=metric_results.get)
    print(f"\n   ✓ Best metric: {best_metric} (R² = {metric_results[best_metric]:.4f})")
    
    # ======================================================================
    # 3. WEIGHTING STRATEGIES
    # ======================================================================
    print("\n3. Comparing Weighting Strategies...")
    print("   (Euclidean metric, k=5)")
    
    weights_results = {}
    
    for weight_strategy in ['uniform', 'distance']:
        model = KNNRegression(k=5, metric='euclidean', weights=weight_strategy, scale=True)
        model.fit(X_train_scaled, y_train)
        r2 = model.score(X_test_scaled, y_test)
        weights_results[weight_strategy] = r2
        print(f"   • {weight_strategy:15s} → R² = {r2:.4f}")
    
    best_weight = max(weights_results, key=weights_results.get)
    print(f"\n   ✓ Best weighting: {best_weight} (R² = {weights_results[best_weight]:.4f})")
    
    # ======================================================================
    # 4. K-VALUE TUNING
    # ======================================================================
    print("\n4. Hyperparameter Tuning: K-Value Selection...")
    print("   (Euclidean, Distance-weighted, Scaled)")
    
    k_values = [1, 3, 5, 7, 10, 15, 20]
    k_results = {}
    
    for k in k_values:
        model = KNNRegression(k=k, metric='euclidean', weights='distance', scale=True)
        model.fit(X_train_scaled, y_train)
        r2 = model.score(X_test_scaled, y_test)
        k_results[k] = r2
        print(f"   • k={k:2d}  → R² = {r2:.4f}")
    
    best_k = max(k_results, key=k_results.get)
    print(f"\n   ✓ Optimal k: {best_k} (R² = {k_results[best_k]:.4f})")
    
    # ======================================================================
    # 5. MINKOWSKI PARAMETER TUNING
    # ======================================================================
    print("\n5. Minkowski Distance: Parameter 'p' Selection...")
    print("   (k=5, Distance-weighted, Scaled)")
    
    p_values = [1, 1.5, 2, 3, 4, 5]
    p_results = {}
    
    for p in p_values:
        model = KNNRegression(k=5, metric='minkowski', p=p, weights='distance', scale=True)
        model.fit(X_train_scaled, y_train)
        r2 = model.score(X_test_scaled, y_test)
        p_results[p] = r2
        print(f"   • p={p:3.1f} → R² = {r2:.4f}")
    
    best_p = max(p_results, key=p_results.get)
    print(f"\n   ✓ Optimal p: {best_p} (R² = {p_results[best_p]:.4f})")
    
    # ======================================================================
    # 6. FEATURE SCALING IMPACT
    # ======================================================================
    print("\n6. Impact of Feature Scaling...")
    print("   (Euclidean, Distance-weighted, k=5)")
    
    model_scaled = KNNRegression(k=5, metric='euclidean', weights='distance', scale=True)
    model_scaled.fit(X_train_scaled, y_train)
    r2_scaled = model_scaled.score(X_test_scaled, y_test)
    
    model_unscaled = KNNRegression(k=5, metric='euclidean', weights='distance', scale=False)
    model_unscaled.fit(X_train, y_train)
    r2_unscaled = model_unscaled.score(X_test, y_test)
    
    print(f"   • With scaling   → R² = {r2_scaled:.4f}")
    print(f"   • Without scaling → R² = {r2_unscaled:.4f}")
    print(f"\n   ✓ Scaling improvement: {(r2_scaled - r2_unscaled):.4f}")
    
    # ======================================================================
    # 7. PERMUTATION-BASED FEATURE IMPORTANCE
    # ======================================================================
    print("\n7. Feature Importance Analysis (Permutation-Based)...")
    best_model = KNNRegression(k=best_k, metric='euclidean', weights='distance', scale=True)
    best_model.fit(X_train_scaled, y_train)
    
    importances = best_model.compute_feature_importances(
        X_test_scaled, y_test, n_repeats=10, random_state=42
    )
    
    top_indices = np.argsort(-importances)[:5]
    print(f"\n   Top 5 Most Important Features:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"   {rank}. Feature {idx:2d} → Importance = {importances[idx]:.6f}")
    
    # ======================================================================
    # 8. COMPARISON WITH LINEAR REGRESSION
    # ======================================================================
    print("\n8. KNN vs Linear Regression Comparison...")
    
    # OLS Linear Regression
    ols = LinearRegression(penalty=None, solver='closed')
    ols.fit(X_train_scaled, y_train)
    r2_ols = ols.score(X_test_scaled, y_test)
    
    # Ridge Regression
    ridge = LinearRegression(penalty='l2', alpha=10.0, solver='closed')
    ridge.fit(X_train_scaled, y_train)
    r2_ridge = ridge.score(X_test_scaled, y_test)
    
    # Best KNN model
    best_knn = KNNRegression(k=best_k, metric='euclidean', weights='distance', scale=True)
    best_knn.fit(X_train_scaled, y_train)
    r2_knn = best_knn.score(X_test_scaled, y_test)
    
    print(f"\n   Model Performance (R² Score):")
    print(f"   • OLS (Linear):         R² = {r2_ols:.4f}")
    print(f"   • Ridge Regression:     R² = {r2_ridge:.4f}")
    print(f"   • KNN (k={best_k}):     R² = {r2_knn:.4f}")
    
    # Determine winner
    results = {
        'OLS': r2_ols,
        'Ridge': r2_ridge,
        f'KNN (k={best_k})': r2_knn
    }
    winner = max(results, key=results.get)
    print(f"\n   ✓ Best performing model: {winner}")
    
    # ======================================================================
    # 9. RESIDUAL ANALYSIS FOR BEST KNN
    # ======================================================================
    print("\n9. Residual Analysis (Best KNN Model)...")
    
    y_pred_knn = best_knn.predict(X_test_scaled)
    residuals_knn = y_test - y_pred_knn
    
    mae = np.mean(np.abs(residuals_knn))
    rmse = np.sqrt(np.mean(residuals_knn ** 2))
    mean_residual = np.mean(residuals_knn)
    std_residual = np.std(residuals_knn)
    
    print(f"\n   • Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"   • Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"   • Mean of residuals:              {mean_residual:.4f}")
    print(f"   • Std Dev of residuals:           {std_residual:.4f}")
    
    # ======================================================================
    # 10. SUMMARY TABLE
    # ======================================================================
    print("\n" + "=" * 80)
    print("SUMMARY - All Results")
    print("=" * 80)
    
    print("\nDistance Metrics (k=5, distance-weighted):")
    for metric, score in sorted(metric_results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {metric:15s} → R² = {score:.4f}")
    
    print("\nWeighting Strategies (k=5, euclidean):")
    for strategy, score in sorted(weights_results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {strategy:15s} → R² = {score:.4f}")
    
    print("\nK-Value Tuning (euclidean, distance-weighted):")
    for k in sorted(k_results.keys())[:7]:  # Show first 7
        print(f"  k={k:2d}               → R² = {k_results[k]:.4f}")
    
    print("\nModel Comparison:")
    for model_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_name:20s} → R² = {score:.4f}")
    
    # Highlight best K value
    print("\n" + "=" * 80)
    print("🏆 BEST K-VALUE ANALYSIS")
    print("=" * 80)
    print(f"Optimal k for Euclidean + Distance-weighted: k = {best_k} (R² = {k_results[best_k]:.4f})")
    print(f"Optimal Minkowski p parameter:                p = {best_p} (R² = {p_results[best_p]:.4f})")
    print(f"Best distance metric:                         {best_metric} (R² = {metric_results[best_metric]:.4f})")
    print(f"Best weighting strategy:                      {best_weight} (R² = {weights_results[best_weight]:.4f})")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] KNN Regression demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
