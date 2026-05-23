"""
Visualization utilities for Linear Regression diagnostics.

Provides text-based and matplotlib-based visualizations for regression analysis.
"""

import numpy as np
from scipy import stats
import warnings

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class RegressionVisualizer:
    """
    Visualization tools for linear regression models.
    
    Provides methods to visualize residuals, predictions, and correlations.
    """
    
    def __init__(self, figsize=(12, 10)):
        """
        Parameters:
        -----------
        figsize : tuple
            Figure size for matplotlib plots (default: (12, 10))
        """
        self.figsize = figsize
        if not HAS_MATPLOTLIB:
            warnings.warn("Matplotlib not installed. Text-based visualizations only.")
    
    def plot_diagnostics(self, model, X, y, save_path=None):
        """
        Create a 2x2 grid of diagnostic plots.
        
        Parameters:
        -----------
        model : LinearRegression
            Fitted regression model
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target values
        save_path : str, optional
            Path to save the figure
        """
        if not HAS_MATPLOTLIB:
            print("ERROR: Matplotlib is required for plotting. Install with: pip install matplotlib")
            return
        
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Linear Regression Diagnostic Plots', fontsize=16, fontweight='bold')
        
        # Plot 1: Residuals vs Fitted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Scale-Location Plot (sqrt of standardized residuals)
        standardized_residuals = residuals / np.std(residuals)
        sqrt_abs_residuals = np.sqrt(np.abs(standardized_residuals))
        axes[1, 0].scatter(y_pred, sqrt_abs_residuals, alpha=0.6, edgecolors='k')
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Standardized Residuals|')
        axes[1, 0].set_title('Scale-Location Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Residuals Histogram
        axes[1, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Histogram of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Diagnostic plots saved to {save_path}")
        
        plt.show()
    
    def plot_actual_vs_predicted(self, model, X, y, save_path=None):
        """
        Plot actual vs predicted values with reference line.
        
        Parameters:
        -----------
        model : LinearRegression
            Fitted regression model
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target values
        save_path : str, optional
            Path to save the figure
        """
        if not HAS_MATPLOTLIB:
            print("ERROR: Matplotlib is required for plotting. Install with: pip install matplotlib")
            return
        
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        y_pred = model.predict(X)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(y, y_pred, alpha=0.6, edgecolors='k', s=50)
        
        # Reference line (y = x)
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals_histogram(self, residuals, save_path=None):
        """
        Plot histogram of residuals with normal distribution overlay.
        
        Parameters:
        -----------
        residuals : array-like
            Residuals from the model
        save_path : str, optional
            Path to save the figure
        """
        if not HAS_MATPLOTLIB:
            print("ERROR: Matplotlib is required for plotting. Install with: pip install matplotlib")
            return
        
        residuals = np.array(residuals)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        counts, bins, patches = ax.hist(residuals, bins=30, density=True, alpha=0.7, 
                                        edgecolor='black', label='Residuals')
        
        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
        
        ax.set_xlabel('Residuals', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_heatmap(self, X, feature_names=None, save_path=None):
        """
        Plot correlation matrix as a heatmap.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        feature_names : list of str, optional
            Names of features
        save_path : str, optional
            Path to save the figure
        """
        if not HAS_MATPLOTLIB:
            print("ERROR: Matplotlib is required for plotting. Install with: pip install matplotlib")
            return
        
        X = np.array(X, dtype=float)
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        corr_matrix = np.corrcoef(X_std.T)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # Set ticks and labels
        n_features = corr_matrix.shape[0]
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(n_features))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticklabels(feature_names)
        
        # Add correlation values
        for i in range(n_features):
            for j in range(n_features):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, label='Correlation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


class TextVisualizer:
    """
    Text-based visualization for environments without matplotlib.
    """
    
    @staticmethod
    def print_residuals_distribution(residuals, bins=20):
        """
        Print ASCII histogram of residuals.
        
        Parameters:
        -----------
        residuals : array-like
            Residuals from the model
        bins : int
            Number of bins (default: 20)
        """
        residuals = np.array(residuals)
        counts, bin_edges = np.histogram(residuals, bins=bins)
        
        print("\n" + "="*70)
        print("RESIDUALS DISTRIBUTION (TEXT-BASED HISTOGRAM)")
        print("="*70)
        
        max_count = np.max(counts)
        for i in range(bins):
            bar_length = int(50 * counts[i] / max_count)
            bar = "█" * bar_length
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            print(f"{bin_center:7.2f} | {bar} {counts[i]:4d}")
    
    @staticmethod
    def print_actual_vs_predicted(y_true, y_pred, max_samples=20):
        """
        Print actual vs predicted values in table format.
        
        Parameters:
        -----------
        y_true : array-like
            Actual values
        y_pred : array-like
            Predicted values
        max_samples : int
            Maximum number of samples to display (default: 20)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        print("\n" + "="*70)
        print("ACTUAL VS PREDICTED VALUES")
        print("="*70)
        print(f"{'Index':<6} {'Actual':<15} {'Predicted':<15} {'Error':<15} {'% Error':<10}")
        print("-"*70)
        
        n_samples = min(max_samples, len(y_true))
        for i in range(n_samples):
            error = y_true[i] - y_pred[i]
            pct_error = 100 * error / y_true[i] if y_true[i] != 0 else 0
            print(f"{i:<6} {y_true[i]:<15.4f} {y_pred[i]:<15.4f} {error:<15.4f} {pct_error:<10.2f}%")
        
        if n_samples < len(y_true):
            print(f"... (showing {n_samples} of {len(y_true)} samples)")
    
    @staticmethod
    def print_correlation_heatmap_text(X, feature_names=None):
        """
        Print ASCII-art correlation heatmap.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        feature_names : list of str, optional
            Names of features
        """
        X = np.array(X, dtype=float)
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        corr_matrix = np.corrcoef(X_std.T)
        
        n_features = corr_matrix.shape[0]
        if feature_names is None:
            feature_names = [f'F{i}' for i in range(n_features)]
        
        print("\n" + "="*70)
        print("CORRELATION MATRIX (TEXT-BASED)")
        print("="*70)
        
        # Header
        print("       ", end="")
        for fname in feature_names:
            print(f"{fname:>8}", end=" ")
        print()
        
        # Rows
        for i, fname in enumerate(feature_names):
            print(f"{fname:>6} ", end="")
            for j in range(n_features):
                val = corr_matrix[i, j]
                if val > 0.7:
                    symbol = "+"
                elif val < -0.7:
                    symbol = "-"
                elif val > 0.3:
                    symbol = "."
                elif val < -0.3:
                    symbol = ":"
                else:
                    symbol = " "
                print(f"{val:7.3f}{symbol} ", end="")
            print()
        
        print("\nKey: + (strong positive)  - (strong negative)  . (moderate positive)  : (moderate negative)")
