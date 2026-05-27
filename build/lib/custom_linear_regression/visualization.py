"""
Visualization utilities for linear regression diagnostics.

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
    """

    def __init__(self, figsize=(12, 10)):
        self.figsize = figsize
        if not HAS_MATPLOTLIB:
            warnings.warn("Matplotlib not installed. Text-based visualizations only.")

    def plot_diagnostics(self, model, X, y, save_path=None):
        if not HAS_MATPLOTLIB:
            print("ERROR: Matplotlib is required for plotting. Install with: pip install matplotlib")
            return

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        y_pred = model.predict(X)
        residuals = y - y_pred

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle("Linear Regression Diagnostic Plots", fontsize=16, fontweight="bold")

        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, edgecolors="k")
        axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=2)
        axes[0, 0].set_xlabel("Fitted Values")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Fitted Values")
        axes[0, 0].grid(True, alpha=0.3)

        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Q-Q Plot")
        axes[0, 1].grid(True, alpha=0.3)

        residual_std = np.std(residuals)
        if np.isclose(residual_std, 0.0):
            standardized_residuals = np.zeros_like(residuals)
        else:
            standardized_residuals = residuals / residual_std
        sqrt_abs_residuals = np.sqrt(np.abs(standardized_residuals))
        axes[1, 0].scatter(y_pred, sqrt_abs_residuals, alpha=0.6, edgecolors="k")
        axes[1, 0].set_xlabel("Fitted Values")
        axes[1, 0].set_ylabel("sqrt(|Standardized Residuals|)")
        axes[1, 0].set_title("Scale-Location Plot")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
        axes[1, 1].set_xlabel("Residuals")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Histogram of Residuals")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Diagnostic plots saved to {save_path}")

        plt.show()

    def plot_actual_vs_predicted(self, model, X, y, save_path=None):
        if not HAS_MATPLOTLIB:
            print("ERROR: Matplotlib is required for plotting. Install with: pip install matplotlib")
            return

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        y_pred = model.predict(X)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y, y_pred, alpha=0.6, edgecolors="k", s=50)

        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.set_title("Actual vs Predicted Values", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_residuals_histogram(self, residuals, save_path=None):
        if not HAS_MATPLOTLIB:
            print("ERROR: Matplotlib is required for plotting. Install with: pip install matplotlib")
            return

        residuals = np.array(residuals, dtype=float)
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            residuals,
            bins=30,
            density=True,
            alpha=0.7,
            edgecolor="black",
            label="Residuals",
        )

        mu, sigma = np.mean(residuals), np.std(residuals)
        sigma = 1.0 if np.isclose(sigma, 0.0) else sigma
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Normal Distribution")

        ax.set_xlabel("Residuals", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Distribution of Residuals", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_correlation_heatmap(self, X, feature_names=None, save_path=None):
        if not HAS_MATPLOTLIB:
            print("ERROR: Matplotlib is required for plotting. Install with: pip install matplotlib")
            return

        X = np.array(X, dtype=float)
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        X_std = (X - means) / stds
        corr_matrix = np.corrcoef(X_std.T)

        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

        n_features = corr_matrix.shape[0]
        if feature_names is None:
            feature_names = [f"Feature_{index}" for index in range(n_features)]

        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(n_features))
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.set_yticklabels(feature_names)

        for row in range(n_features):
            for col in range(n_features):
                ax.text(
                    col,
                    row,
                    f"{corr_matrix[row, col]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        fig.colorbar(image, ax=ax, label="Correlation")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()


class TextVisualizer:
    """
    Text-based visualization for environments without matplotlib.
    """

    @staticmethod
    def print_residuals_distribution(residuals, bins=20):
        residuals = np.array(residuals, dtype=float)
        counts, bin_edges = np.histogram(residuals, bins=bins)

        print("\n" + "=" * 70)
        print("RESIDUALS DISTRIBUTION (TEXT-BASED HISTOGRAM)")
        print("=" * 70)

        max_count = max(np.max(counts), 1)
        for index in range(bins):
            bar_length = int(50 * counts[index] / max_count)
            bar = "#" * bar_length
            bin_center = (bin_edges[index] + bin_edges[index + 1]) / 2
            print(f"{bin_center:7.2f} | {bar} {counts[index]:4d}")

    @staticmethod
    def print_actual_vs_predicted(y_true, y_pred, max_samples=20):
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        print("\n" + "=" * 70)
        print("ACTUAL VS PREDICTED VALUES")
        print("=" * 70)
        print(f"{'Index':<6} {'Actual':<15} {'Predicted':<15} {'Error':<15} {'% Error':<10}")
        print("-" * 70)

        n_samples = min(max_samples, len(y_true))
        for index in range(n_samples):
            error = y_true[index] - y_pred[index]
            pct_error = 100 * error / y_true[index] if y_true[index] != 0 else 0
            print(
                f"{index:<6} {y_true[index]:<15.4f} {y_pred[index]:<15.4f} "
                f"{error:<15.4f} {pct_error:<10.2f}%"
            )

        if n_samples < len(y_true):
            print(f"... (showing {n_samples} of {len(y_true)} samples)")

    @staticmethod
    def print_correlation_heatmap_text(X, feature_names=None):
        X = np.array(X, dtype=float)
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        X_std = (X - means) / stds
        corr_matrix = np.corrcoef(X_std.T)

        n_features = corr_matrix.shape[0]
        if feature_names is None:
            feature_names = [f"F{index}" for index in range(n_features)]

        print("\n" + "=" * 70)
        print("CORRELATION MATRIX (TEXT-BASED)")
        print("=" * 70)

        print("       ", end="")
        for feature_name in feature_names:
            print(f"{feature_name:>8}", end=" ")
        print()

        for row_index, feature_name in enumerate(feature_names):
            print(f"{feature_name:>6} ", end="")
            for col_index in range(n_features):
                value = corr_matrix[row_index, col_index]
                if value > 0.7:
                    symbol = "+"
                elif value < -0.7:
                    symbol = "-"
                elif value > 0.3:
                    symbol = "."
                elif value < -0.3:
                    symbol = ":"
                else:
                    symbol = " "
                print(f"{value:7.3f}{symbol} ", end="")
            print()

        print("\nKey: + (strong positive)  - (strong negative)  . (moderate positive)  : (moderate negative)")