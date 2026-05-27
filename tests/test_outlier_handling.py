"""
Test script demonstrating outlier detection and handling in LinearRegression.
"""

import numpy as np

from diagnostics import OutlierDetector
from linear_regression import LinearRegression


def create_data_with_outliers():
    np.random.seed(42)

    X_train = np.random.randn(120, 3) * 5 + 20
    X_test = np.random.randn(40, 3) * 5 + 20

    true_weights = np.array([2.5, -1.2, 0.7])
    y_train = np.dot(X_train, true_weights) + np.random.randn(120) * 2
    y_test = np.dot(X_test, true_weights) + np.random.randn(40) * 2

    outlier_rows = [5, 11, 32, 77]
    X_train[outlier_rows, 0] *= 8
    X_train[outlier_rows, 2] *= -4
    y_train[outlier_rows] += np.array([180.0, -220.0, 140.0, -160.0])

    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = create_data_with_outliers()

    print("=" * 70)
    print("Testing Outlier Detection and Handling")
    print("=" * 70)

    detector = OutlierDetector(strategy="zscore", threshold=3.0)
    detection = detector.detect(X_train)
    print(f"Detected outlier rows: {detection['n_outlier_rows']}")
    print(f"Per-feature outlier counts: {detection['feature_outlier_counts']}")

    configs = [
        {"name": "No outlier handling", "kwargs": {}},
        {
            "name": "Remove outliers (zscore)",
            "kwargs": {
                "outlier_strategy": "zscore",
                "outlier_threshold": 3.0,
                "outlier_action": "remove",
            },
        },
        {
            "name": "Clip outliers (iqr)",
            "kwargs": {
                "outlier_strategy": "iqr",
                "outlier_threshold": 1.5,
                "outlier_action": "clip",
            },
        },
    ]

    for config in configs:
        print("\n" + "-" * 70)
        print(config["name"])
        print("-" * 70)

        model = LinearRegression(penalty=None, solver="closed", **config["kwargs"])
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"R^2 Score: {score:.4f}")
        print(f"Weights: {model.weights}")
        print(f"Bias: {model.bias:.4f}")


if __name__ == "__main__":
    main()
