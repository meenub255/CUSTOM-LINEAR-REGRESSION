#!/usr/bin/env python3
"""
Example showing PCA usage with regression models.
"""

import numpy as np
import pandas as pd
from custom_linear_regression import PCA, LinearRegression, KNNRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def main():
    print("PCA + REGRESSION EXAMPLE")
    print("=" * 50)
    
    # Load and preprocess the Thrissur house prices dataset
    print("Loading Thrissur house prices dataset...")
    df = pd.read_csv("data/thrissur_house_prices.csv")
    
    # Select numeric features for regression
    df_clean = df.copy()
    
    # Convert Avg_Price_Per_SqFt from "4.09k/sq.ft" format to numeric
    def convert_price_per_sqft(val):
        if isinstance(val, str):
            if 'k' in val:
                return float(val.replace('k', '').replace('/sq.ft', '').strip()) * 1000
            else:
                return float(val.replace('/sq.ft', '').strip())
        return float(val)
    
    df_clean['Avg_Price_Per_SqFt_Numeric'] = df_clean['Avg_Price_Per_SqFt'].apply(convert_price_per_sqft)
    
    # Select features and target
    feature_cols = ['BHK', 'Area_SqFt', 'Avg_Price_Per_SqFt_Numeric']
    target_col = 'Price_Lacs'
    
    # Remove rows with missing values
    df_clean = df_clean[feature_cols + [target_col]].dropna()
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # ========== BASELINE: Raw Features ==========
    print("\n" + "="*50)
    print("BASELINE: Raw Features")
    print("="*50)
    
    # Linear Regression
    lr_raw = LinearRegression()
    lr_raw.fit(X_train, y_train)
    lr_raw_pred = lr_raw.predict(X_test)
    lr_raw_r2 = r2_score(y_test, lr_raw_pred)
    print(f"Linear Regression R²: {lr_raw_r2:.4f}")
    
    # KNN Regression
    knn_raw = KNNRegression(k=5, metric='euclidean', weights='distance', scale=True)
    knn_raw.fit(X_train, y_train)
    knn_raw_pred = knn_raw.predict(X_test)
    knn_raw_r2 = r2_score(y_test, knn_raw_pred)
    print(f"KNN Regression R²:    {knn_raw_r2:.4f}")
    
    # ========== WITH PCA ==========
    print("\n" + "="*50)
    print("WITH PCA (95% Variance Retained)")
    print("="*50)
    
    # Apply PCA to retain 95% of variance
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"PCA features:      {X_train_pca.shape[1]}")
    print(f"Variance retained: {np.sum(pca.explained_variance_ratio_):.2%}")
    print(f"Components: {pca.n_components_}")
    
    # Show what PCA learned
    print(f"\nPCA Components (feature loadings):")
    for i, (comp, var_ratio) in enumerate(zip(pca.components_, pca.explained_variance_ratio_)):
        print(f"  PC{i+1} ({var_ratio:.1%} variance):")
        for j, (feat, loading) in enumerate(zip(feature_cols, comp)):
            print(f"    {feat}: {loading:+.3f}")
    
    # Linear Regression with PCA
    lr_pca = LinearRegression()
    lr_pca.fit(X_train_pca, y_train)
    lr_pca_pred = lr_pca.predict(X_test_pca)
    lr_pca_r2 = r2_score(y_test, lr_pca_pred)
    print(f"\nLinear Regression + PCA R²: {lr_pca_r2:.4f}")
    
    # KNN Regression with PCA
    knn_pca = KNNRegression(k=5, metric='euclidean', weights='distance', scale=True)
    knn_pca.fit(X_train_pca, y_train)
    knn_pca_pred = knn_pca.predict(X_test_pca)
    knn_pca_r2 = r2_score(y_test, knn_pca_pred)
    print(f"KNN Regression + PCA R²:    {knn_pca_r2:.4f}")
    
    # ========== COMPARISON ==========
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Method':<25} {'Raw Features':<15} {'With PCA':<15} {'Change':<10}")
    print("-" * 65)
    print(f"{'Linear Regression':<25} {lr_raw_r2:<15.4f} {lr_pca_r2:<15.4f} {lr_pca_r2-lr_raw_r2:<+10.4f}")
    print(f"{'KNN Regression':<25} {knn_raw_r2:<15.4f} {knn_pca_r2:<15.4f} {knn_pca_r2-knn_raw_r2:<+10.4f}")
    
    # Show reconstruction error
    print("\n" + "="*50)
    print("PCA RECONSTRUCTION QUALITY")
    print("="*50)
    X_train_reconstructed = pca.inverse_transform(X_train_pca)
    reconstruction_error = np.mean((X_train - X_train_reconstructed) ** 2)
    print(f"Reconstruction MSE: {reconstruction_error:.4f}")
    print(f"This shows how well we can approximate original data from PCA components")
    
    print("\n" + "="*50)
    print("KEY INSIGHTS")
    print("="*50)
    print("1. PCA successfully reduced dimensionality while preserving most variance")
    print("2. The learned components show how original features combine")
    print("3. Regression performance is maintained (or sometimes improved)")
    print("4. PCA helps with multicollinearity and can reduce overfitting")
    print("5. The custom library now offers a complete ML workflow:")
    print("   - Data preprocessing (missing values, outliers)")
    print("   - Dimensionality reduction (PCA)")
    print("   - Feature selection (Forward/Backward)")
    print("   - Regression models (Linear, KNN)")
    print("   - Evaluation and visualization tools")

if __name__ == "__main__":
    main()