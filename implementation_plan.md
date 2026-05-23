# Implementation Plan - Extending Custom Linear Regression Library

This plan outlines the design and implementation of significant enhancements to our custom NumPy-based `LinearRegression` library. These features will extend the library to support robust regression, explainable AI, advanced feature selection, and high-fidelity CLI visualizations.

## User Review Required

> [!NOTE]
> All visualizations (Correlation Matrix Heatmap and Actual vs. Predicted Scatter Plot) will be built using pure Python and NumPy, rendering directly in the terminal using ANSI escape color codes. No heavy external plotting libraries (like matplotlib or seaborn) will be required, keeping our library ultra-lightweight and dependency-free.

> [!IMPORTANT]
> The new feature selection algorithms (Forward Selection and Backward Elimination based on correlation) will run step-wise. 
> - **Forward Selection** will iteratively select the feature that has the highest absolute correlation with the current model's residuals.
> - **Backward Elimination** will iteratively drop the feature whose removal results in the smallest drop in model $R^2$ performance.

## Proposed Changes

We will modify two key files and add a helper script to test all the new capabilities.

### 1. Robust & Feature-Rich Library

#### [MODIFY] [linear_regression.py](file:///d:/linear%20regression/linear_regression.py)
*   **Huber Loss Support**: Add `loss='huber'` and a tuning parameter `delta=1.35`. We will calculate the Huber loss gradients mathematically in the Gradient Descent loop.
*   **Scikit-Learn API Alignment**: Add `@property` decorators for `.coef_` (returning weights) and `.intercept_` (returning bias) for seamless integration with existing ML workflows.
*   **Sensitivity Simulation**: Add a `.simulate_sensitivity(X, perturbation=0.01)` method. This will perturb each feature column and measure the mean absolute change in predictions, providing a clean sensitivity/importance score for each feature.
*   **Feature Selection**: Add two robust algorithms:
    - `.correlation_forward_selection(X, y, n_features)`: Employs an iterative residual correlation approach to select features step-by-step.
    - `.correlation_backward_elimination(X, y, n_features)`: Iteratively trains models, dropping the least significant feature that has the lowest impact on $R^2$.

### 2. CLI Visualizations & Utilities

#### [NEW] [visualization_utils.py](file:///d:/linear%20regression/visualization_utils.py)
*   **ANSI Correlation Heatmap (`show_correlation_map`)**: A pure NumPy function that computes the Pearson correlation matrix and prints a stunning, colored grid inside the terminal using ANSI escape codes (green/blue for positive, red/magenta for negative, gray for neutral).
*   **ASCII Actual vs. Predicted Plot (`show_actual_vs_predicted`)**: A function that prints a beautiful 20x50 ASCII scatter plot in the CLI comparing actual vs. predicted values, drawing a reference line `y = x` using dots (`.`) and data points using stars (`*`).

### 3. Comprehensive Testing

#### [MODIFY] [example_usage.py](file:///d:/linear%20regression/example_usage.py)
*   Update the example workflow to load the house price data.
*   Display the **ANSI Correlation Heatmap** in the terminal prior to model training.
*   Perform **Forward Selection** to select the top features.
*   Train the robust **Huber Loss** model.
*   Run the **Sensitivity Simulation** and display feature importance in the terminal.
*   Print the **ASCII Actual vs. Predicted Plot** to visually inspect model fit.

---

## Verification Plan

### Automated Manual Verification
*   We will run the `example_usage.py` script.
*   We will visually verify that the ANSI correlation map renders with correct color gradients.
*   We will verify that the ASCII scatter plot maps predicted values accurately on the diagonal.
*   We will verify that the sensitivity simulation identifies `Area_SqFt` as the most sensitive feature.
*   We will verify that Scikit-like properties `.coef_` and `.intercept_` are retrievable.
