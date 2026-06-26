import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

DATA_PATH = Path("/home/niranjan/ai_ids_project/data/processed/processed_data.csv")
LABEL_CLASSES_PATH = Path("/home/niranjan/ai_ids_project/data/processed/label_classes.npy")
SAVE_DIR = Path("/home/niranjan/ai_ids_project/models/saved")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Total rows: {len(df):,}, Columns: {df.shape[1]}")

feature_cols = [c for c in df.columns if c != "Label"]
X = df[feature_cols].values
y = df["Label"].values

label_classes = np.load(LABEL_CLASSES_PATH, allow_pickle=True)
print(f"  Classes ({len(label_classes)}): {label_classes.tolist()}")

# Stratified split: 70% train, 15% val, 15% test
print("\nSplitting data (70/15/15 stratified)...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# Train
print("\nTraining RandomForestClassifier (n_estimators=200, n_jobs=-1)...")
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, verbose=1)
rf.fit(X_train, y_train)

# Validate
print("\nValidation set performance:")
y_val_pred = rf.predict(X_val)
print(f"  Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")

# Test evaluation
print("\nTest set evaluation:")
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"  Overall Accuracy: {acc:.4f}")

print("\nPer-class metrics:")
present_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
present_labels = [label_classes[i] for i in present_classes]

report = classification_report(
    y_test, y_pred,
    labels=present_classes,
    target_names=present_labels,
    digits=4,
    zero_division=0
)
print(report)

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=present_classes)
header = f"{'':>30}" + "".join(f"{label_classes[i]:>10}" for i in present_classes)
print(header)
for i, row in zip(present_classes, cm):
    print(f"{label_classes[i]:>30}" + "".join(f"{v:>10}" for v in row))

# Save model
model_path = SAVE_DIR / "rf_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(rf, f)
print(f"\nModel saved to: {model_path}")
print(f"Label classes:  {LABEL_CLASSES_PATH}")
print("\nRF Training Complete")
