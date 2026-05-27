#!/usr/bin/env python3
"""
Evaluation script for Linear Regression and KNN Regression on Thrissur house prices dataset.
Shows complete metrics including K values tried and best K value for KNN regression.
"""

import pandas as pd
import numpy as np
from custom_linear_regression import LinearRegression, KNNRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the Thrissur house prices dataset."""
    print("Loading Thrissur house prices dataset...")
    df = pd.read_csv("thrissur_house_prices.csv")
    
    # Select numeric features for regression
    # We'll use BHK, Area_SqFt, and Avg_Price_Per_SqFt (converted from string)
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
    # Features: BHK, Area_SqFt, Avg_Price_Per_SqFt_Numeric
    # Target: Price_Lacs
    feature_cols = ['BHK', 'Area_SqFt', 'Avg_Price_Per_SqFt_Numeric']
    target_col = 'Price_Lacs'
    
    # Remove rows with missing values
    df_clean = df_clean[feature_cols + [target_col]].dropna()
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    print(f"Price range: {y.min():.2f} - {y.max():.2f} lakhs")
    
    return X, y, feature_cols

def evaluate_linear_regression(X, y):
    """Evaluate Linear Regression model."""
    print("\n" + "="*60)
    print("LINEAR REGRESSION EVALUATION")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    lr_model = LinearRegression(
        fit_intercept=True,
        penalty=None,  # OLS
        solver='gd',
        lr=0.01,
        n_iters=1000
    )
    
    lr_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score:  {test_r2:.4f}")
    print(f"Training RMSE:     {train_rmse:.4f}")
    print(f"Testing RMSE:      {test_rmse:.4f}")
    print(f"Training MAE:      {train_mae:.4f}")
    print(f"Testing MAE:       {test_mae:.4f}")
    
    print(f"\nModel Coefficients:")
    for i, coef in enumerate(lr_model.coef_):
        print(f"  Feature {i} ({['BHK', 'Area_SqFt', 'Avg_Price_Per_SqFt'][i]}): {coef:.4f}")
    print(f"  Intercept: {lr_model.intercept_:.4f}")
    
    return lr_model, {
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae
    }

def evaluate_knn_regression(X, y):
    """Evaluate KNN Regression model with K value analysis."""
    print("\n" + "="*60)
    print("K-NEAREST NEIGHBORS REGRESSION EVALUATION")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different K values
    k_values = list(range(1, 21))  # K from 1 to 20
    k_scores = []
    
    print("Testing different K values:")
    print("-" * 40)
    
    for k in k_values:
        knn_model = KNNRegression(
            k=k,
            metric='euclidean',
            weights='distance',
            scale=True
        )
        
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        score = r2_score(y_test, y_pred)
        k_scores.append((k, score))
        print(f"K={k:2d}: R² Score = {score:.4f}")
    
    # Find best K
    best_k, best_score = max(k_scores, key=lambda x: x[1])
    
    print("-" * 40)
    print(f"Best K value: {best_k} (R² Score: {best_score:.4f})")
    
    # Train final model with best K
    print(f"\nTraining final KNN model with K={best_k}...")
    knn_model_final = KNNRegression(
        k=best_k,
        metric='euclidean',
        weights='distance',
        scale=True
    )
    
    knn_model_final.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = knn_model_final.predict(X_train)
    y_pred_test = knn_model_final.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\nFinal Model Performance (K={best_k}):")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score:  {test_r2:.4f}")
    print(f"Training RMSE:     {train_rmse:.4f}")
    print(f"Testing RMSE:      {test_rmse:.4f}")
    print(f"Training MAE:      {train_mae:.4f}")
    print(f"Testing MAE:       {test_mae:.4f}")
    
    # Feature importance
    print(f"\nFeature Importance (Permutation-based):")
    importances = knn_model_final.compute_feature_importances(X_test, y_test)
    feature_names = ['BHK', 'Area_SqFt', 'Avg_Price_Per_SqFt']
    for name, importance in zip(feature_names, importances):
        print(f"  {name}: {importance:.4f}")
    
    return knn_model_final, {
        'best_k': best_k,
        'best_score': best_score,
        'k_scores': k_scores,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mae': train_mae, 'test_mae': test_mae
    }

def main():
    """Main evaluation function."""
    print("THRISSUR HOUSE PRICES REGRESSION ANALYSIS")
    print("=" * 60)
    
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data()
    
    # Evaluate Linear Regression
    lr_model, lr_metrics = evaluate_linear_regression(X, y)
    
    # Evaluate KNN Regression
    knn_model, knn_metrics = evaluate_knn_regression(X, y)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Linear Regression - Test R²:  {lr_metrics['test_r2']:.4f}")
    print(f"KNN Regression    - Test R²:  {knn_metrics['test_r2']:.4f} (K={knn_metrics['best_k']})")
    
    if knn_metrics['test_r2'] > lr_metrics['test_r2']:
        improvement = ((knn_metrics['test_r2'] - lr_metrics['test_r2']) / lr_metrics['test_r2']) * 100
        print(f"KNN outperforms Linear Regression by {improvement:.2f}%")
    else:
        improvement = ((lr_metrics['test_r2'] - knn_metrics['test_r2']) / knn_metrics['test_r2']) * 100
        print(f"Linear Regression outperforms KNN by {improvement:.2f}%")

if __name__ == "__main__":
    main()