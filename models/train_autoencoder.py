import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

DATA_PATH = Path("/home/niranjan/ai_ids_project/data/processed/processed_data.csv")
SAVE_DIR = Path("/home/niranjan/ai_ids_project/models/saved")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Total rows: {len(df):,}, Columns: {df.shape[1]}")

feature_cols = [c for c in df.columns if c != "Label"]
X_all = df[feature_cols].values.astype(np.float32)
y_all = df["Label"].values

# Filter benign for training
X_benign = X_all[y_all == 0]
print(f"  Benign rows for training: {len(X_benign):,}")
print(f"  Feature dimensions: {X_benign.shape[1]}")

# Build autoencoder
input_dim = X_benign.shape[1]  # 77

inputs = layers.Input(shape=(input_dim,))
encoded = layers.Dense(64, activation="relu")(inputs)
encoded = layers.Dense(32, activation="relu")(encoded)
decoded = layers.Dense(64, activation="relu")(encoded)
decoded = layers.Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = models.Model(inputs, decoded, name="autoencoder")
autoencoder.compile(optimizer="adam", loss="mse")

print("\nModel summary:")
autoencoder.summary()

# Train
print("\nTraining autoencoder on benign traffic...")
history = autoencoder.fit(
    X_benign, X_benign,
    epochs=20,
    batch_size=256,
    validation_split=0.1,
    shuffle=True,
    verbose=1,
)

# Compute reconstruction errors on benign data
print("\nComputing reconstruction errors...")
benign_reconstructions = autoencoder.predict(X_benign, batch_size=1024, verbose=0)
benign_mse = np.mean(np.square(X_benign - benign_reconstructions), axis=1)

threshold = float(np.mean(benign_mse) + 2 * np.std(benign_mse))
print(f"  Benign MSE  — mean: {np.mean(benign_mse):.6f}, std: {np.std(benign_mse):.6f}")
print(f"  Anomaly threshold (mean + 2*std): {threshold:.6f}")

# Evaluate on full dataset
all_reconstructions = autoencoder.predict(X_all, batch_size=1024, verbose=0)
all_mse = np.mean(np.square(X_all - all_reconstructions), axis=1)
predictions = (all_mse > threshold).astype(int)

print("\nDetection summary (threshold-based):")
for label_id in np.unique(y_all):
    mask = y_all == label_id
    detected = predictions[mask].sum()
    total = mask.sum()
    print(f"  Label {label_id:2d}: {detected:>7,} / {total:>7,} flagged as anomaly ({100*detected/total:.1f}%)")

# Save
model_path = SAVE_DIR / "autoencoder.h5"
threshold_path = SAVE_DIR / "threshold.npy"

autoencoder.save(model_path)
np.save(threshold_path, np.array(threshold))

print(f"\nModel saved to:     {model_path}")
print(f"Threshold saved to: {threshold_path}")
print(f"\nFinal threshold: {threshold:.6f}")
print("Done.")
