"""
Custom Linear Regression Library

A NumPy-based linear regression library with feature selection,
diagnostics, visualization, KNN regression, PCA, and decision tree.
"""

__version__ = "0.2.0"

from .linear_regression import LinearRegression
from .knn_regression import KNNRegression
from .feature_selection import ForwardSelection, BackwardElimination
from .diagnostics import (
    HeteroscedasticityTest,
    HeteroscedacityTest,
    MulticollinearityTest,
    NormalityTest,
    OutlierDetector,
)
from .visualization import RegressionVisualizer, TextVisualizer
from .pca import PCA
from .decision_tree import DecisionTreeRegressor
from .exceptions import (
    CustomLinearRegressionError,
    NotFittedError,
    OptimizationFailedError,
    DataQualityError,
)

__all__ = [
    # Version
    "__version__",
    # Main models
    "LinearRegression",
    "KNNRegression",
    "DecisionTreeRegressor",
    # Feature selection
    "ForwardSelection",
    "BackwardElimination",
    # Diagnostics
    "NormalityTest",
    "MulticollinearityTest",
    "HeteroscedasticityTest",
    "HeteroscedacityTest",
    "OutlierDetector",
    # Dimensionality reduction
    "PCA",
    # Visualization
    "RegressionVisualizer",
    "TextVisualizer",
    # Exceptions
    "CustomLinearRegressionError",
    "NotFittedError",
    "OptimizationFailedError",
    "DataQualityError",
]