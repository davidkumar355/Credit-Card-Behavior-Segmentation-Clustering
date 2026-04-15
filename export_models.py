import pandas as pd
import numpy as np
import joblib
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

print("Starting model training and export pipeline...")

# Load Data
df_raw = pd.read_csv("credit_card_data.csv")
df = df_raw.copy()

# 1. Imputation
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True)
df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median(), inplace=True)

# 2. Ratio Engineering
df['SPEND_INTENSITY'] = df['PURCHASES'] / (df['CREDIT_LIMIT'] + 1)
df['PAYMENT_DISCIPLINE'] = df['PAYMENTS'] / (df['MINIMUM_PAYMENTS'] + 1)
df['REVOLVING_BEHAVIOR'] = df['BALANCE'] / (df['CREDIT_LIMIT'] + 1)
df['CASH_DEPENDENCY'] = df['CASH_ADVANCE'] / (df['PURCHASES'] + 1)

features = [
    'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 
    'INSTALLMENTS_PURCHASES', 'ONEOFF_PURCHASES_FREQUENCY', 
    'PRC_FULL_PAYMENT', 'TENURE', 'SPEND_INTENSITY', 
    'PAYMENT_DISCIPLINE', 'REVOLVING_BEHAVIOR', 'CASH_DEPENDENCY'
]

df_features = df[features].copy()
df_transformed = pd.DataFrame(index=df_features.index)

# 3. Skewness treatment (with memory to persist max values for inference)
reflection_max_vals = {}

for col in features:
    capped = winsorize(df_features[col], limits=(0.01, 0.01))
    
    if col in ['BALANCE_FREQUENCY', 'TENURE']:
        max_val = np.max(capped)
        reflection_max_vals[col] = max_val
        df_transformed[col] = np.log1p(max_val + 1 - capped)
    else:
        df_transformed[col] = np.log1p(capped)

# 4. Fit Models
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_transformed)

pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_pca)

# Ensure output directory exists
os.makedirs("model", exist_ok=True)

# 5. Export models via joblib
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(pca, "model/pca.pkl")
joblib.dump(kmeans, "model/kmeans_model.pkl")

# Note: We also need to save the reflection_max_vals so the Streamlit app can use them exactly.
joblib.dump(reflection_max_vals, "model/reflection_max_vals.pkl")

print("✅ Success! Models exported to the 'model' directory:")
print("- model/scaler.pkl")
print("- model/pca.pkl")
print("- model/kmeans_model.pkl")
print("- model/reflection_max_vals.pkl")
