import pandas as pd
import numpy as np

# Load existing data
df = pd.read_csv('thrissur_house_prices.csv')

# Extract unique localities
localities = df['Locality'].unique()
city = 'Thrissur'

# Define target distribution based on original data
# BHK typically ranges from 2 to 6
# Area typically ranges from 800 to 4000
# Price has a positive correlation with Area and BHK

np.random.seed(42)
new_rows = []
n_new_samples = 200

# Base price per sqft per locality to maintain realistic data
locality_base_price = df.groupby('Locality')['Price_Lacs'].mean() / df.groupby('Locality')['Area_SqFt'].mean()
overall_base_price = (df['Price_Lacs'] / df['Area_SqFt']).mean()

for _ in range(n_new_samples):
    bhk = np.random.choice([2, 3, 4, 5], p=[0.3, 0.4, 0.2, 0.1])
    
    # Base area depending on BHK
    area_mean = bhk * 600
    area = int(np.random.normal(loc=area_mean, scale=area_mean*0.2))
    area = max(800, min(area, 6000))  # Clip between 800 and 6000 sqft
    
    locality = np.random.choice(localities)
    
    # Determine price based on locality and area, adding some noise
    price_per_sqft = locality_base_price.get(locality, overall_base_price)
    
    # Introduce random variation in price (-20% to +20%)
    price = area * price_per_sqft * np.random.uniform(0.8, 1.2)
    price_lacs = round(price, 1)
    
    new_rows.append({
        'BHK': bhk,
        'Area_SqFt': area,
        'City': city,
        'Price_Lacs': price_lacs,
        'Locality': locality
    })

# Append new data to CSV
new_df = pd.DataFrame(new_rows)
updated_df = pd.concat([df, new_df], ignore_index=True)

# Save to CSV
updated_df.to_csv('thrissur_house_prices.csv', index=False)
print(f"Added {n_new_samples} new synthetic samples.")
print(f"Total samples now: {len(updated_df)}")
