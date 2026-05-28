#!/usr/bin/env python3
"""
Example showing Decision Tree usage with regression models.
"""

import numpy as np
import pandas as pd
from custom_linear_regression import DecisionTreeRegressor, LinearRegression, KNNRegression, PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def main():
    print("DECISION TREE REGRESSION EXAMPLE")
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
    
    # ========== BASELINE MODELS ==========
    print("\n" + "="*50)
    print("BASELINE MODELS")
    print("="*50)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    print(f"Linear Regression R²: {lr_r2:.4f}")
    
    # KNN Regression
    knn = KNNRegression(k=5, metric='euclidean', weights='distance', scale=True)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_r2 = r2_score(y_test, knn_pred)
    print(f"KNN Regression R²:    {knn_r2:.4f}")
    
    # ========== DECISION TREE MODEL ==========
    print("\n" + "="*50)
    print("DECISION TREE REGRESSION")
    print("="*50)
    
    # Decision Tree with default parameters
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_r2 = r2_score(y_test, dt_pred)
    print(f"Decision Tree R²:     {dt_r2:.4f}")
    print(f"Tree Depth: {dt.max_depth_}")
    print(f"Number of Leaves: {dt.n_leaves_}")
    
    # Feature importance
    print(f"\nFeature Importances:")
    for feat, imp in zip(feature_cols, dt.feature_importances_):
        print(f"  {feat}: {imp:.4f}")
    
    # ========== TUNED DECISION TREE ==========
    print("\n" + "="*50)
    print("TUNED DECISION TREE")
    print("="*50)
    
    # Try different parameters to avoid overfitting/underfitting
    dt_tuned = DecisionTreeRegressor(
        max_depth=5,      # Limit depth to prevent overfitting
        min_samples_split=10,  # Require more samples to split
        min_samples_leaf=5     # Require more samples in leaves
    )
    dt_tuned.fit(X_train, y_train)
    dt_tuned_pred = dt_tuned.predict(X_test)
    dt_tuned_r2 = r2_score(y_test, dt_tuned_pred)
    print(f"Tuned Decision Tree R²: {dt_tuned_r2:.4f}")
    print(f"Tree Depth: {dt_tuned.max_depth_}")
    print(f"Number of Leaves: {dt_tuned.n_leaves_}")
    
    # Feature importance for tuned tree
    print(f"\nTuned Feature Importances:")
    for feat, imp in zip(feature_cols, dt_tuned.feature_importances_):
        print(f"  {feat}: {imp:.4f}")
    
    # ========== DECISION TREE WITH PCA ==========
    print("\n" + "="*50)
    print("DECISION TREE + PCA")
    print("="*50)
    
    # Apply PCA to reduce dimensionality and noise
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"PCA features:      {X_train_pca.shape[1]}")
    print(f"Variance retained: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    # Decision Tree on PCA-transformed data
    dt_pca = DecisionTreeRegressor(max_depth=5)
    dt_pca.fit(X_train_pca, y_train)
    dt_pca_pred = dt_pca.predict(X_test_pca)
    dt_pca_r2 = r2_score(y_test, dt_pca_pred)
    print(f"\nDecision Tree + PCA R²: {dt_pca_r2:.4f}")
    print(f"Tree Depth: {dt_pca.max_depth_}")
    print(f"Number of Leaves: {dt_pca.n_leaves_}")
    
    # Show what PCA learned
    print(f"\nPCA Components (Feature Loadings):")
    for i, (comp, var_ratio) in enumerate(zip(pca.components_, pca.explained_variance_ratio_)):
        print(f"  PC{i+1} ({var_ratio:.1%} variance):")
        for j, (feat, loading) in enumerate(zip(feature_cols, comp)):
            print(f"    {feat}: {loading:+.3f}")
    
    # ========== DECISION TREE WITH FEATURE SELECTION ==========
    print("\n" + "="*50)
    print("DECISION TREE + FEATURE SELECTION")
    print("="*50)
    
    # Use Forward Selection to find best feature subset for Decision Tree
    from custom_linear_regression import ForwardSelection
    
    # Use a simple model for feature selection (can use Decision Tree itself)
    base_model_for_selection = DecisionTreeRegressor(max_depth=3)
    selector = ForwardSelection(base_model_for_selection, k_features=3)  # We have 3 features
    selector.fit(X_train, y_train)
    
    selected_features = selector.k_feature_idx_
    print(f"Selected features: {[feature_cols[i] for i in selected_features]}")
    
    # Train final model with selected features
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    dt_fs = DecisionTreeRegressor(max_depth=5)
    dt_fs.fit(X_train_selected, y_train)
    dt_fs_pred = dt_fs.predict(X_test_selected)
    dt_fs_r2 = r2_score(y_test, dt_fs_pred)
    print(f"\nDecision Tree + Feature Selection R²: {dt_fs_r2:.4f}")
    
    # ========== COMPARISON SUMMARY ==========
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Model':<30} {'R² Score':<12} {'Notes'}")
    print("-" * 60)
    print(f"{'Linear Regression':<30} {lr_r2:<12.4f} {'Baseline linear model'}")
    print(f"{'KNN Regression':<30} {knn_r2:<12.4f} {'K=5, euclidean, distance-weighted'}")
    print(f"{'Decision Tree':<30} {dt_r2:<12.4f} {'Default parameters'}")
    print(f"{'Tuned Decision Tree':<30} {dt_tuned_r2:<12.4f} {'max_depth=5, min_samples_split=10'}")
    print(f"{'Decision Tree + PCA':<30} {dt_pca_r2:<12.4f} {'PCA for dimensionality reduction'}")
    print(f"{'Decision Tree + FS':<30} {dt_fs_r2:<12.4f} {'Feature Selection'}")
    
    # Find best model
    models = [
        ("Linear Regression", lr_r2),
        ("KNN Regression", knn_r2),
        ("Decision Tree", dt_r2),
        ("Tuned Decision Tree", dt_tuned_r2),
        ("Decision Tree + PCA", dt_pca_r2),
        ("Decision Tree + FS", dt_fs_r2)
    ]
    
    best_model_name, best_model_score = max(models, key=lambda x: x[1])
    print(f"\n🏆 Best Model: {best_model_name} (R² = {best_model_score:.4f})")
    
    # ========== DECISION TREE INTERPRETATION ==========
    print("\n" + "="*50)
    print("DECISION TREE INTERPRETATION")
    print("="*50)
    
    # Show some decision rules from the tree (simplified)
    def print_tree_rules(tree, feature_names, depth=0):
        indent = "  " * depth
        if "value" in tree:
            print(f"{indent}→ Predict: {tree['value']:.2f}")
            return
        
        feature_idx = tree["feature_index"]
        threshold = tree["threshold"]
        feature_name = feature_names[feature_idx]
        
        print(f"{indent}if {feature_name} <= {threshold:.2f}:")
        print_tree_rules(tree["left"], feature_names, depth + 1)
        print(f"{indent}else:  # {feature_name} > {threshold:.2f}")
        print_tree_rules(tree["right"], feature_names, depth + 1)
    
    print("Decision Tree Rules (first few levels):")
    print_tree_rules(dt_tuned.tree_, feature_cols)
    
    print("\n" + "="*50)
    print("KEY INSIGHTS")
    print("="*50)
    print("1. Decision Trees can capture non-linear relationships that Linear Regression misses")
    print("2. They provide interpretable rules (if-then statements) for business understanding")
    print("3. Feature importance helps identify which variables drive predictions")
    print("4. Can be combined with PCA for dimensionality reduction and noise filtering")
    print("5. Feature Selection can improve performance by removing irrelevant variables")
    print("6. Hyperparameter tuning (max_depth, min_samples_*) prevents overfitting")
    print("7. The custom library now offers a complete supervised learning toolkit:")
    print("   - Linear Models (OLS, Lasso, Ridge)")
    print("   - Instance-Based Learning (KNN)")
    print("   - Tree-Based Models (Decision Tree)")
    print("   - Dimensionality Reduction (PCA)")
    print("   - Feature Selection (Forward/Backward)")
    print("   - Statistical Diagnostics")
    print("   - Visualization Tools")

if __name__ == "__main__":
    main()