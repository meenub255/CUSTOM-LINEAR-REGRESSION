import numpy as np
import csv
from custom_linear_regression import (
    LinearRegression,
    RegressionVisualizer,
    TextVisualizer,
)

def r2_score_manual(y_true, y_pred):
    """Calculate R^2 manually to score exponentiated predictions."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def load_and_preprocess_data(filename):
    """Load data and compute basic features."""
    raw_data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                bhk = float(row['BHK'])
                area = float(row['Area_SqFt'])
                price = float(row['Price_Lacs'])
                locality = row['Locality'].strip()
                possession = row['Possession_Status'].strip()
                
                # Filter out absurd outliers manually first
                if price <= 0 or area <= 0:
                    continue
                
                # Basic Feature Engineering
                ready_to_move = 1.0 if possession == 'Ready to move' else 0.0
                price_per_sqft = (price * 100000) / area
                
                raw_data.append({
                    'bhk': bhk,
                    'area': area,
                    'ready_to_move': ready_to_move,
                    'locality': locality,
                    'price': price,
                    'price_per_sqft': price_per_sqft
                })
            except (ValueError, KeyError):
                continue
    return raw_data

def train_test_split(data, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    np.random.seed(random_state)
    indices = np.random.permutation(len(data))
    test_samples = int(len(data) * test_size)
    
    test_data = [data[i] for i in indices[:test_samples]]
    train_data = [data[i] for i in indices[test_samples:]]
    
    return train_data, test_data

def feature_engineering_and_encoding(train_data, test_data):
    """
    Apply advanced feature engineering:
    1. Target encode locality
    2. Polynomial and Interaction terms
    3. Logarithmic transformation of the target
    """
    # 1. Target Encoding for Locality (computed on training data only)
    locality_sums = {}
    locality_counts = {}
    total_sum = 0
    total_count = 0
    
    for row in train_data:
        loc = row['locality']
        pps = row['price_per_sqft']
        locality_sums[loc] = locality_sums.get(loc, 0) + pps
        locality_counts[loc] = locality_counts.get(loc, 0) + 1
        total_sum += pps
        total_count += 1
        
    global_mean = total_sum / total_count if total_count > 0 else 0
    locality_means = {loc: locality_sums[loc] / locality_counts[loc] for loc in locality_sums}
    
    def extract_features_and_target(dataset):
        X = []
        y_true = []
        y_log = []
        for row in dataset:
            loc_mean = locality_means.get(row['locality'], global_mean)
            area = row['area']
            bhk = row['bhk']
            
            # Interaction & Polynomial features
            # Interaction & Polynomial features
            area_squared = area ** 2
            interaction_area_bhk = area * bhk
            
            features = [
                bhk, 
                area, 
                row['ready_to_move'], 
                loc_mean, 
                area_squared, 
                interaction_area_bhk
            ]
            
            X.append(features)
            y_true.append(row['price'])
            y_log.append(np.log(row['price'])) # Target for the model
            
        return np.array(X), np.array(y_true), np.array(y_log)

    X_train, y_train_true, y_train_log = extract_features_and_target(train_data)
    X_test, y_test_true, y_test_log = extract_features_and_target(test_data)
    
    return X_train, y_train_true, y_train_log, X_test, y_test_true, y_test_log

def standardize_features(X_train, X_test):
    """Standardize features (zero mean, unit variance)."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled, mean, std

def main():
    print("=" * 80)
    print("ADVANCED LINEAR REGRESSION (LOG-TRANSFORM + POLYNOMIALS)")
    print("=" * 80)
    
    # 1. Load Data
    raw_data = load_and_preprocess_data("thrissur_house_prices.csv")
    print(f"\n1. Loaded {len(raw_data)} valid samples.")
    
    # 2. Split
    train_data, test_data = train_test_split(raw_data, test_size=0.2)
    print(f"2. Split into {len(train_data)} training and {len(test_data)} testing samples.")
    
    # 3. Feature Engineering & Target Extraction
    print("3. Applying Target Encoding, Polynomial Features, and Log(Price)...")
    X_train, y_train_true, y_train_log, X_test, y_test_true, y_test_log = feature_engineering_and_encoding(train_data, test_data)
    
    # 4. Standardize
    print("4. Standardizing features...")
    X_train_scaled, X_test_scaled, mean, std = standardize_features(X_train, X_test)
    
    # 5. Train Model
    print("\n5. Training Custom Linear Regression Model on LOG-Prices...")
    # Ridge with strong regularization helps prevent overfitting on the new interaction terms
    model = LinearRegression(
        fit_intercept=True,
        penalty='l2', 
        alpha=10.0, 
        solver='closed',
        outlier_strategy='iqr', 
        outlier_threshold=2.5
    )
    
    model.fit(X_train_scaled, y_train_log)
    
    # 6. Evaluate
    # Predict in log space, then exponentiate to get real prices
    log_pred_train = model.predict(X_train_scaled)
    log_pred_test = model.predict(X_test_scaled)
    
    pred_train_true = np.exp(log_pred_train)
    pred_test_true = np.exp(log_pred_test)
    
    r2_train = r2_score_manual(y_train_true, pred_train_true)
    r2_test = r2_score_manual(y_test_true, pred_test_true)
    
    print("\n" + "=" * 40)
    print("MODEL PERFORMANCE (On Real Prices)")
    print("=" * 40)
    print(f"Training R^2 Score : {r2_train:.4f}")
    print(f"Testing R^2 Score  : {r2_test:.4f}")
    
    if r2_test > 0.6:
        print("\n[SUCCESS] Amazing! Log-transforming the prices and adding interactions")
        print("   pushed the testing R^2 score beyond 0.60!")
    elif r2_test > 0.5:
        print("\n[SUCCESS] Good improvement! Testing R^2 score is steady.")
    else:
        print("\n[WARNING] The R^2 score dropped. The model might be overfitting.")
        
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE (Standardized Coefficients)")
    print("=" * 50)
    features = ['BHK', 'Area_SqFt', 'Ready_To_Move', 'Locality_Mean_Price_SqFt', 'Area_Squared', 'Area_x_BHK']
    for f, c in zip(features, model.coef_):
        print(f"{f:30}: {c:.4f}")
    print(f"{'Intercept (Log Space)':30}: {model.intercept_:.4f}")
    
    print("\n" + "=" * 70)
    print("ACTUAL VS PREDICTED (Test Set - Real Prices)")
    print("=" * 70)
    TextVisualizer.print_actual_vs_predicted(y_test_true[:15], pred_test_true[:15])

if __name__ == "__main__":
    main()
