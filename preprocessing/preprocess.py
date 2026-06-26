import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pathlib import Path
import os

RAW_DIR = Path("/home/niranjan/ai_ids_project/data/raw")
OUT_DIR = Path("/home/niranjan/ai_ids_project/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

parquet_files = list(RAW_DIR.glob("*.parquet"))
print(f"Found {len(parquet_files)} files:")
for f in parquet_files:
    print(f"  {f.name}")

dfs = []
for f in parquet_files:
    print(f"\nReading {f.name}...")
    df = pd.read_parquet(f)
    print(f"  Shape: {df.shape}")
    dfs.append(df)

print("\nCombining all files...")
combined = pd.concat(dfs, ignore_index=True)
print(f"Total records: {len(combined):,}")
print(f"Columns: {list(combined.columns)}")

# Clean column names
combined.columns = combined.columns.str.strip()

# Find label column (could be 'Label' or 'label')
label_col = None
for col in combined.columns:
    if col.lower() == 'label':
        label_col = col
        break

if label_col is None:
    print("ERROR: No label column found!")
    print("Available columns:", list(combined.columns))
    exit(1)

print(f"\nLabel column found: '{label_col}'")
print("\nClass distribution:")
print(combined[label_col].value_counts())

# Rename to standard 'Label'
combined.rename(columns={label_col: 'Label'}, inplace=True)

# Separate features and label
labels = combined['Label'].copy()
features = combined.drop(columns=['Label'])

# Keep only numeric columns
features = features.select_dtypes(include=[np.number])
print(f"\nNumeric features: {features.shape[1]}")

# Replace inf values
features.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill nulls with 0
features.fillna(0, inplace=True)

# Scale features
print("Scaling features...")
scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)
features_scaled = pd.DataFrame(scaled, columns=features.columns)

# Encode labels
print("Encoding labels...")
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
features_scaled['Label'] = encoded_labels

# Save
out_path = OUT_DIR / "processed_data.csv"
print(f"\nSaving to {out_path}...")
features_scaled.to_csv(out_path, index=False)

print("\n✅ Done!")
print(f"Final shape: {features_scaled.shape}")
print(f"Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Save label encoder classes
np.save(OUT_DIR / "label_classes.npy", le.classes_)
print("Label classes saved.")
