"""
Test script demonstrating normality, multicollinearity, heteroscedasticity diagnostics and visualizations.
"""

import numpy as np
from linear_regression import LinearRegression
from diagnostics import NormalityTest, MulticollinearityTest, HeteroscedacityTest
from visualization import RegressionVisualizer, TextVisualizer


def test_normality_with_normal_residuals():
    """Test normality with normally distributed residuals."""
    print("\n" + "="*70)
    print("TEST 1: NORMALITY WITH NORMAL RESIDUALS")
    print("="*70)
    
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3) * 10 + 50
    true_weights = np.array([2.0, 3.0, -1.5])
    
    # Generate residuals from normal distribution
    residuals = np.random.normal(0, 5, n_samples)
    y = np.dot(X, true_weights) + residuals
    
    model = LinearRegression(penalty=None, solver='closed')
    model.fit(X, y)
    
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    print(f"\nResidual Statistics:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std Dev: {np.std(residuals):.6f}")
    print(f"  Min: {np.min(residuals):.6f}")
    print(f"  Max: {np.max(residuals):.6f}")
    
    normality_test = NormalityTest(alpha=0.05)
    normality_test.run_all_tests(residuals)
    normality_test.print_summary()


def test_normality_with_skewed_residuals():
    """Test normality with non-normal (skewed) residuals."""
    print("\n" + "="*70)
    print("TEST 2: NORMALITY WITH SKEWED RESIDUALS")
    print("="*70)
    
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3) * 10 + 50
    true_weights = np.array([2.0, 3.0, -1.5])
    
    # Generate skewed residuals
    residuals = np.random.exponential(5, n_samples) - 5  # Skewed distribution
    y = np.dot(X, true_weights) + residuals
    
    model = LinearRegression(penalty=None, solver='closed')
    model.fit(X, y)
    
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    print(f"\nResidual Statistics:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std Dev: {np.std(residuals):.6f}")
    print(f"  Min: {np.min(residuals):.6f}")
    print(f"  Max: {np.max(residuals):.6f}")
    
    normality_test = NormalityTest(alpha=0.05)
    normality_test.run_all_tests(residuals)
    normality_test.print_summary()


def test_multicollinearity_low():
    """Test multicollinearity with low correlation features."""
    print("\n" + "="*70)
    print("TEST 3: MULTICOLLINEARITY - LOW CORRELATION (Acceptable)")
    print("="*70)
    
    np.random.seed(42)
    n_samples = 100
    
    # Create independent features
    X = np.random.randn(n_samples, 4) * 10 + 50
    
    print(f"\nDataset: {n_samples} samples, {X.shape[1]} features")
    
    multicollinearity_test = MulticollinearityTest(vif_threshold=5.0)
    vif_scores = multicollinearity_test.calculate_vif(X)
    
    multicollinearity_test.print_vif_summary()
    
    corr_matrix = multicollinearity_test.get_correlation_matrix(X)
    print("\nCorrelation Matrix:")
    print(corr_matrix)


def test_multicollinearity_high():
    """Test multicollinearity with highly correlated features."""
    print("\n" + "="*70)
    print("TEST 4: MULTICOLLINEARITY - HIGH CORRELATION (Problematic)")
    print("="*70)
    
    np.random.seed(42)
    n_samples = 100
    
    # Create highly correlated features
    feature1 = np.random.randn(n_samples) * 10 + 50
    feature2 = feature1 + np.random.normal(0, 1, n_samples)  # Highly correlated with feature1
    feature3 = feature1 * 2 + np.random.normal(0, 2, n_samples)  # Also correlated with feature1
    feature4 = np.random.randn(n_samples) * 10 + 50  # Independent
    
    X = np.column_stack([feature1, feature2, feature3, feature4])
    
    print(f"\nDataset: {n_samples} samples, {X.shape[1]} features")
    print("Note: Features 0, 1, 2 are highly correlated; Feature 3 is independent")
    
    multicollinearity_test = MulticollinearityTest(vif_threshold=5.0)
    vif_scores = multicollinearity_test.calculate_vif(X)
    
    multicollinearity_test.print_vif_summary()
    
    corr_matrix = multicollinearity_test.get_correlation_matrix(X)
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    high_pairs = multicollinearity_test.detect_high_correlation_pairs(X, threshold=0.8)
    print(f"\nHigh Correlation Pairs (threshold=0.8):")
    for feat_i, feat_j, corr in high_pairs:
        print(f"  {feat_i} <-> {feat_j}: {corr:.4f}")


def test_integrated_diagnostics():
    """Test integrated diagnostics with a complete regression model."""
    print("\n" + "="*70)
    print("TEST 5: INTEGRATED DIAGNOSTICS")
    print("="*70)
    
    np.random.seed(42)
    n_samples = 100
    
    # Create moderately correlated features
    X = np.random.randn(n_samples, 4) * 10 + 50
    X[:, 1] = X[:, 0] + np.random.normal(0, 3, n_samples)  # Moderate correlation
    
    true_weights = np.array([2.0, 1.5, -1.0, 0.5])
    y = np.dot(X, true_weights) + np.random.normal(0, 5, n_samples)
    
    print(f"\nDataset: {n_samples} samples, {X.shape[1]} features")
    
    # Fit model
    model = LinearRegression(penalty='l2', alpha=1.0, solver='closed')
    model.fit(X, y)
    
    print(f"Model fitted successfully")
    print(f"  Weights: {model.weights}")
    print(f"  Bias: {model.bias:.4f}")
    print(f"  R² Score (on training): {model.score(X, y):.4f}")
    
    # Calculate residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Normality Test
    print("\n--- NORMALITY DIAGNOSTICS ---")
    normality_test = NormalityTest(alpha=0.05)
    normality_test.run_all_tests(residuals)
    normality_test.print_summary()
    
    # Multicollinearity Test
    print("\n--- MULTICOLLINEARITY DIAGNOSTICS ---")
    multicollinearity_test = MulticollinearityTest(vif_threshold=5.0)
    multicollinearity_test.run_all_tests(X)
    multicollinearity_test.print_vif_summary()
    
    corr_matrix = multicollinearity_test.get_correlation_matrix(X)
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Heteroscedasticity Test
    print("\n--- HETEROSCEDASTICITY DIAGNOSTICS ---")
    heteroscedasticity_test = HeteroscedacityTest(alpha=0.05)
    heteroscedasticity_test.run_all_tests(residuals, X)
    heteroscedasticity_test.print_summary()
    
    # Text-based Visualizations
    print("\n--- TEXT-BASED VISUALIZATIONS ---")
    TextVisualizer.print_residuals_distribution(residuals)
    TextVisualizer.print_actual_vs_predicted(y, y_pred, max_samples=15)
    TextVisualizer.print_correlation_heatmap_text(X)


def test_with_heteroscedasticity():
    """Test with heteroscedastic data (non-constant variance)."""
    print("\n" + "="*70)
    print("TEST 6: HETEROSCEDASTICITY (Non-constant variance)")
    print("="*70)
    
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3) * 10 + 50
    true_weights = np.array([2.0, 1.5, -1.0])
    
    # Generate residuals with heteroscedasticity (variance increases with X)
    residuals = np.random.normal(0, X[:, 0] / 10, n_samples)  # Variance depends on X
    y = np.dot(X, true_weights) + residuals
    
    model = LinearRegression(penalty=None, solver='closed')
    model.fit(X, y)
    
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    print(f"\nDataset: {n_samples} samples, {X.shape[1]} features")
    print("Note: Variance increases with X values (heteroscedastic)")
    
    # Test heteroscedasticity
    heteroscedasticity_test = HeteroscedacityTest(alpha=0.05)
    heteroscedasticity_test.run_all_tests(residuals, X)
    heteroscedasticity_test.print_summary()
    
    # Visualization
    print("\n--- VISUALIZATION ---")
    TextVisualizer.print_actual_vs_predicted(y, y_pred, max_samples=15)


if __name__ == "__main__":
    test_normality_with_normal_residuals()
    test_normality_with_skewed_residuals()
    test_multicollinearity_low()
    test_multicollinearity_high()
    test_integrated_diagnostics()
    test_with_heteroscedasticity()
    
    print("\n" + "="*70)
    print("All diagnostic tests completed!")
    print("="*70)
    print("\nNote: To generate matplotlib plots, use RegressionVisualizer:")
    print("  visualizer = RegressionVisualizer()")
    print("  visualizer.plot_diagnostics(model, X, y)")
    print("  visualizer.plot_actual_vs_predicted(model, X, y)")
