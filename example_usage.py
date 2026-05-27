import numpy as np
import csv
from linear_regression import LinearRegression
from feature_selection import ForwardSelection, BackwardElimination

def load_data(filename):
    X_base = []
    y = []
    localities = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            try:
                # Base Features: BHK (col 0), Area_SqFt (col 1)
                features = [float(row[0]), float(row[1])]
                # Target: Price_Lacs (col 3)
                target = float(row[3])
                # Locality (col 4)
                locality = row[4].strip()
                
                X_base.append(features)
                y.append(target)
                localities.append(locality)
            except (ValueError, IndexError):
                # Handle any rows with missing or malformed data
                continue
                
    # One-Hot Encode the Locality column
    unique_localities = list(set(localities))
    unique_localities.sort() # Sort to keep the order deterministic
    
    X = []
    for i in range(len(X_base)):
        # Create a zero vector for localities
        one_hot = [0.0] * len(unique_localities)
        # Set the corresponding locality index to 1.0
        loc_index = unique_localities.index(localities[i])
        one_hot[loc_index] = 1.0
        
        # Combine base features (BHK, Area) with one-hot encoded locality
        full_features = X_base[i] + one_hot
        X.append(full_features)
        
    return np.array(X), np.array(y)

def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_samples = int(len(X) * test_size)
    
    X_train = X[indices[test_samples:]]
    y_train = y[indices[test_samples:]]
    X_test = X[indices[:test_samples]]
    y_test = y[indices[:test_samples]]
    
    return X_train, X_test, y_train, y_test

def StandardScaler_fit_transform(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Handle zero division
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std

def StandardScaler_transform(X, mean, std):
    return (X - mean) / std

def main():
    print("Loading dataset...")
    X, y = load_data("thrissur_house_prices.csv")
    
    print(f"Dataset size: {len(X)} samples")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Standardize features (highly recommended for Gradient Descent and Regularization)
    X_train_scaled, mean, std = StandardScaler_fit_transform(X_train)
    X_test_scaled = StandardScaler_transform(X_test, mean, std)
    
    # 1. Ordinary Least Squares (OLS) - Closed Form
    print("\n--- Ordinary Least Squares (OLS) ---")
    ols = LinearRegression(penalty=None, solver='closed')
    ols.fit(X_train_scaled, y_train)
    r2_ols = ols.score(X_test_scaled, y_test)
    print(f"Weights: {ols.weights}")
    print(f"Bias: {ols.bias}")
    print(f"Test R^2 Score: {r2_ols:.4f}")
    
    # 2. Ridge Regression (L2) - Closed Form
    print("\n--- Ridge Regression (L2) ---")
    ridge = LinearRegression(penalty='l2', alpha=10.0, solver='closed')
    ridge.fit(X_train_scaled, y_train)
    r2_ridge = ridge.score(X_test_scaled, y_test)
    print(f"Weights: {ridge.weights}")
    print(f"Bias: {ridge.bias}")
    print(f"Test R^2 Score: {r2_ridge:.4f}")

    # 3. Lasso Regression (L1) - Gradient Descent
    print("\n--- Lasso Regression (L1) (MSE Loss) ---")
    lasso = LinearRegression(penalty='l1', alpha=10.0, lr=0.01, n_iters=5000, solver='gd', loss='mse')
    lasso.fit(X_train_scaled, y_train)
    r2_lasso = lasso.score(X_test_scaled, y_test)
    print(f"Weights: {lasso.weights}")
    print(f"Bias: {lasso.bias}")
    print(f"Test R^2 Score: {r2_lasso:.4f}")

    # 4. Ordinary Least Squares (OLS) - Gradient Descent with MAE (Robust to Outliers)
    print("\n--- OLS Gradient Descent with MAE Loss (Robust) ---")
    ols_mae = LinearRegression(penalty=None, lr=0.1, n_iters=5000, solver='gd', loss='mae')
    ols_mae.fit(X_train_scaled, y_train)
    r2_ols_mae = ols_mae.score(X_test_scaled, y_test)
    print(f"Weights: {ols_mae.weights}")
    print(f"Bias: {ols_mae.bias}")
    print(f"Test R^2 Score: {r2_ols_mae:.4f}")

    # 5. Forward Selection
    print("\n--- Forward Selection ---")
    print(f"Original number of features: {X_train_scaled.shape[1]}")
    fwd_sel = ForwardSelection(penalty=None, cv_splits=5)
    fwd_sel.fit(X_train_scaled, y_train, max_features=8)
    selected_fwd = fwd_sel.get_selected_features()
    print(f"Selected features (indices): {selected_fwd}")
    print(f"Number of selected features: {len(selected_fwd)}")
    X_train_fwd = fwd_sel.transform(X_train_scaled)
    X_test_fwd = fwd_sel.transform(X_test_scaled)
    model_fwd = LinearRegression(penalty=None, solver='closed')
    model_fwd.fit(X_train_fwd, y_train)
    r2_fwd = model_fwd.score(X_test_fwd, y_test)
    print(f"Test R^2 Score with selected features: {r2_fwd:.4f}")

    # 6. Backward Elimination
    print("\n--- Backward Elimination ---")
    print(f"Original number of features: {X_train_scaled.shape[1]}")
    bwd_elim = BackwardElimination(penalty=None, cv_splits=5)
    bwd_elim.fit(X_train_scaled, y_train, min_features=3)
    selected_bwd = bwd_elim.get_selected_features()
    print(f"Selected features (indices): {selected_bwd}")
    print(f"Number of selected features: {len(selected_bwd)}")
    X_train_bwd = bwd_elim.transform(X_train_scaled)
    X_test_bwd = bwd_elim.transform(X_test_scaled)
    model_bwd = LinearRegression(penalty=None, solver='closed')
    model_bwd.fit(X_train_bwd, y_train)
    r2_bwd = model_bwd.score(X_test_bwd, y_test)
    print(f"Test R^2 Score with selected features: {r2_bwd:.4f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Model Comparison")
    print("="*60)
    print(f"OLS (All Features):             R^2 = {r2_ols:.4f}")
    print(f"Ridge (All Features):           R^2 = {r2_ridge:.4f}")
    print(f"Lasso (All Features):           R^2 = {r2_lasso:.4f}")
    print(f"OLS MAE (All Features):         R^2 = {r2_ols_mae:.4f}")
    print(f"Forward Selection:              R^2 = {r2_fwd:.4f}")
    print(f"Backward Elimination:           R^2 = {r2_bwd:.4f}")

if __name__ == "__main__":
    main()
