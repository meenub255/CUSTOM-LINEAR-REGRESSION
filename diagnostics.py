"""
Diagnostic tools for Linear Regression: Normality and Multicollinearity tests.

This module provides comprehensive statistical tests and diagnostics for checking
linear regression assumptions.
"""

import numpy as np
from scipy import stats


class NormalityTest:
    """
    Test for normality of residuals using multiple statistical tests.
    
    Linear regression assumes that residuals are normally distributed.
    This class provides multiple tests to verify this assumption.
    """
    
    def __init__(self, alpha=0.05):
        """
        Parameters:
        -----------
        alpha : float
            Significance level for hypothesis tests (default: 0.05)
        """
        self.alpha = alpha
        self.results = {}
    
    def shapiro_wilk_test(self, residuals):
        """
        Shapiro-Wilk Test for normality.
        
        Parameters:
        -----------
        residuals : array-like
            Residuals from the regression model
            
        Returns:
        --------
        result : dict
            Contains test statistic, p-value, and interpretation
        """
        residuals = np.array(residuals)
        statistic, p_value = stats.shapiro(residuals)
        
        result = {
            'test': 'Shapiro-Wilk',
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > self.alpha,
            'interpretation': 'Normal' if p_value > self.alpha else 'Not Normal'
        }
        self.results['shapiro_wilk'] = result
        return result
    
    def anderson_darling_test(self, residuals):
        """
        Anderson-Darling Test for normality.
        
        Parameters:
        -----------
        residuals : array-like
            Residuals from the regression model
            
        Returns:
        --------
        result : dict
            Contains test statistic, critical value, and interpretation
        """
        residuals = np.array(residuals)
        result_obj = stats.anderson(residuals, dist='norm')
        
        # Use 5% significance level (index 2)
        critical_value = result_obj.critical_values[2]
        is_normal = result_obj.statistic < critical_value
        
        result = {
            'test': 'Anderson-Darling',
            'statistic': result_obj.statistic,
            'critical_value': critical_value,
            'is_normal': is_normal,
            'interpretation': 'Normal' if is_normal else 'Not Normal'
        }
        self.results['anderson_darling'] = result
        return result
    
    def jarque_bera_test(self, residuals):
        """
        Jarque-Bera Test for normality.
        
        Tests if skewness and excess kurtosis match a normal distribution.
        
        Parameters:
        -----------
        residuals : array-like
            Residuals from the regression model
            
        Returns:
        --------
        result : dict
            Contains test statistic, p-value, skewness, kurtosis, and interpretation
        """
        residuals = np.array(residuals)
        statistic, p_value = stats.jarque_bera(residuals)
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)
        
        result = {
            'test': 'Jarque-Bera',
            'statistic': statistic,
            'p_value': p_value,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal': p_value > self.alpha,
            'interpretation': 'Normal' if p_value > self.alpha else 'Not Normal'
        }
        self.results['jarque_bera'] = result
        return result
    
    def kolmogorov_smirnov_test(self, residuals):
        """
        Kolmogorov-Smirnov Test for normality.
        
        Compares residuals to a standard normal distribution.
        
        Parameters:
        -----------
        residuals : array-like
            Residuals from the regression model
            
        Returns:
        --------
        result : dict
            Contains test statistic, p-value, and interpretation
        """
        residuals = np.array(residuals)
        # Standardize residuals
        residuals_std = (residuals - np.mean(residuals)) / np.std(residuals)
        statistic, p_value = stats.ks_2samp(residuals_std, np.random.normal(0, 1, len(residuals_std)))
        
        result = {
            'test': 'Kolmogorov-Smirnov',
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > self.alpha,
            'interpretation': 'Normal' if p_value > self.alpha else 'Not Normal'
        }
        self.results['kolmogorov_smirnov'] = result
        return result
    
    def run_all_tests(self, residuals):
        """
        Run all normality tests.
        
        Parameters:
        -----------
        residuals : array-like
            Residuals from the regression model
            
        Returns:
        --------
        results : dict
            Dictionary containing results from all tests
        """
        self.shapiro_wilk_test(residuals)
        self.anderson_darling_test(residuals)
        self.jarque_bera_test(residuals)
        self.kolmogorov_smirnov_test(residuals)
        return self.results
    
    def print_summary(self):
        """Print a summary of all normality test results."""
        print("\n" + "="*70)
        print("NORMALITY TEST RESULTS")
        print("="*70)
        
        for test_name, result in self.results.items():
            print(f"\n{result['test']}:")
            print(f"  Statistic: {result['statistic']:.4f}")
            if 'p_value' in result:
                print(f"  P-value: {result['p_value']:.6f}")
            if 'critical_value' in result:
                print(f"  Critical Value: {result['critical_value']:.4f}")
            if 'skewness' in result:
                print(f"  Skewness: {result['skewness']:.4f}")
                print(f"  Kurtosis: {result['kurtosis']:.4f}")
            print(f"  Result: {result['interpretation']}")


class MulticollinearityTest:
    """
    Test for multicollinearity among features.
    
    Multicollinearity occurs when features are highly correlated, which can
    lead to unstable coefficient estimates and inflated standard errors.
    """
    
    def __init__(self, vif_threshold=5.0):
        """
        Parameters:
        -----------
        vif_threshold : float
            VIF threshold for detecting multicollinearity (default: 5.0)
            VIF > 5 is generally considered problematic
        """
        self.vif_threshold = vif_threshold
        self.correlation_matrix = None
        self.vif_scores = None
        self.feature_names = None
    
    def calculate_vif(self, X, feature_names=None):
        """
        Calculate Variance Inflation Factor (VIF) for each feature.
        
        VIF measures how much the variance of a regression coefficient is inflated
        due to multicollinearity. VIF > 5 is typically considered problematic.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        feature_names : list of str, optional
            Names of features for better readability
            
        Returns:
        --------
        vif_dict : dict
            Dictionary with feature indices/names and their VIF scores
        """
        X = np.array(X, dtype=float)
        n_features = X.shape[1]
        
        # Standardize features
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        # Calculate correlation matrix
        self.correlation_matrix = np.corrcoef(X_std.T)
        
        vif_dict = {}
        self.vif_scores = {}
        
        for i in range(n_features):
            # Calculate R² for regressing feature i on all other features
            X_others = np.delete(X_std, i, axis=1)
            y = X_std[:, i]
            
            # Add intercept
            X_reg = np.c_[np.ones(X_others.shape[0]), X_others]
            
            # Calculate R²
            try:
                beta = np.linalg.lstsq(X_reg, y, rcond=None)[0]
                y_pred = np.dot(X_reg, beta)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
            except:
                r_squared = 0
            
            # Calculate VIF
            if r_squared >= 1.0:
                vif = np.inf
            else:
                vif = 1 / (1 - r_squared)
            
            feature_label = feature_names[i] if feature_names else f"Feature_{i}"
            vif_dict[feature_label] = vif
            self.vif_scores[feature_label] = vif
        
        return vif_dict
    
    def get_correlation_matrix(self, X):
        """
        Calculate Pearson correlation matrix.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
            
        Returns:
        --------
        corr_matrix : array of shape (n_features, n_features)
            Correlation matrix
        """
        X = np.array(X, dtype=float)
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        self.correlation_matrix = np.corrcoef(X_std.T)
        return self.correlation_matrix
    
    def detect_high_correlation_pairs(self, X, threshold=0.9):
        """
        Detect pairs of features with high correlation.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        threshold : float
            Correlation threshold to flag as problematic (default: 0.9)
            
        Returns:
        --------
        high_corr_pairs : list of tuples
            List of (feature_i, feature_j, correlation) with high correlation
        """
        if self.correlation_matrix is None:
            self.get_correlation_matrix(X)
        
        high_corr_pairs = []
        n_features = self.correlation_matrix.shape[0]
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = abs(self.correlation_matrix[i, j])
                if corr > threshold:
                    high_corr_pairs.append((f"Feature_{i}", f"Feature_{j}", corr))
        
        return high_corr_pairs
    
    def print_vif_summary(self):
        """Print a summary of VIF scores."""
        print("\n" + "="*70)
        print("VARIANCE INFLATION FACTOR (VIF) RESULTS")
        print("="*70)
        print(f"VIF Threshold: {self.vif_threshold}")
        print(f"(VIF > {self.vif_threshold} indicates problematic multicollinearity)")
        print("-"*70)
        
        if self.vif_scores:
            for feature, vif in self.vif_scores.items():
                status = "⚠️  PROBLEMATIC" if vif > self.vif_threshold else "✓ OK"
                if vif == np.inf:
                    print(f"{feature:20s}: INF (Perfect multicollinearity) {status}")
                else:
                    print(f"{feature:20s}: {vif:8.4f} {status}")
    
    def print_correlation_summary(self, threshold=0.9):
        """Print a summary of high correlations."""
        print("\n" + "="*70)
        print("HIGH CORRELATION PAIRS")
        print("="*70)
        print(f"Threshold: {threshold}")
        print("-"*70)
        
        high_pairs = self.detect_high_correlation_pairs(
            np.eye(self.correlation_matrix.shape[0]) 
            if self.correlation_matrix is not None 
            else None, 
            threshold=threshold
        )
        
        if high_pairs:
            for feat_i, feat_j, corr in high_pairs:
                print(f"{feat_i} <-> {feat_j}: {corr:.4f}")
        else:
            print("No high correlation pairs detected.")
    
    def run_all_tests(self, X, feature_names=None, corr_threshold=0.9):
        """
        Run all multicollinearity tests.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        feature_names : list of str, optional
            Names of features
        corr_threshold : float
            Correlation threshold for high correlation detection
            
        Returns:
        --------
        results : dict
            Dictionary containing VIF scores and correlation matrix
        """
        self.feature_names = feature_names
        self.calculate_vif(X, feature_names)
        self.get_correlation_matrix(X)
        
        return {
            'vif_scores': self.vif_scores,
            'correlation_matrix': self.correlation_matrix,
            'high_correlation_pairs': self.detect_high_correlation_pairs(X, corr_threshold)
        }
