import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_PATH = Path("/home/niranjan/ai_ids_project/data/processed/processed_data.csv")
LABEL_CLASSES_PATH = Path("/home/niranjan/ai_ids_project/data/processed/label_classes.npy")
SAVE_DIR = Path("/home/niranjan/ai_ids_project/models/saved")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Total rows: {len(df):,}, Columns: {df.shape[1]}")

feature_cols = [c for c in df.columns if c != "Label"]
X_full = df[feature_cols].values.astype(np.float32)
y_full = df["Label"].values.astype(np.int64)

label_classes = np.load(LABEL_CLASSES_PATH, allow_pickle=True)

# 10% stratified sample
print("\nSampling 10% of data (stratified)...")
X, _, y, _ = train_test_split(
    X_full, y_full, test_size=0.90, stratify=y_full, random_state=42
)
print(f"  Sampled rows: {len(X):,}")

# Train IsolationForest
print("\nTraining IsolationForest (n_estimators=100, contamination=0.05)...")
iso = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
iso.fit(X)

# Predict: -1 = anomaly, 1 = normal
print("\nPredicting anomalies...")
raw_preds = iso.predict(X)

# Convert: -1 → 1 (anomaly), 1 → 0 (normal)
y_pred_binary = np.where(raw_preds == -1, 1, 0)

# Binary true labels: 0 = benign, 1 = attack
y_true_binary = np.where(y == 0, 0, 1)

# Per-class detection rate
print("\nDetection rate per class:")
print(f"  {'Class':<35} {'Flagged':>8} {'Total':>8} {'Rate':>8}")
print(f"  {'-'*63}")
for label_id in sorted(np.unique(y)):
    mask = y == label_id
    flagged = y_pred_binary[mask].sum()
    total = mask.sum()
    rate = flagged / total * 100
    class_name = label_classes[label_id]
    print(f"  {class_name:<35} {flagged:>8,} {total:>8,} {rate:>7.1f}%")

# Overall binary metrics
precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
recall    = recall_score(y_true_binary, y_pred_binary, zero_division=0)
f1        = f1_score(y_true_binary, y_pred_binary, zero_division=0)
total_attacks  = y_true_binary.sum()
total_flagged  = y_pred_binary.sum()
true_positives = ((y_pred_binary == 1) & (y_true_binary == 1)).sum()
false_positives = ((y_pred_binary == 1) & (y_true_binary == 0)).sum()

print(f"\nOverall binary metrics (anomaly = attack):")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"\n  Total attacks in sample : {total_attacks:,}")
print(f"  Total flagged as anomaly: {total_flagged:,}")
print(f"  True positives          : {true_positives:,}")
print(f"  False positives         : {false_positives:,}")

# Save model
model_path = SAVE_DIR / "isolation_forest.pkl"
with open(model_path, "wb") as f:
    pickle.dump(iso, f)
print(f"\nModel saved to: {model_path}")
print("\nIsolation Forest Training Complete")
