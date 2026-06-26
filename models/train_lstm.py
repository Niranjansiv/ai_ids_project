import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

DATA_PATH = Path("/home/niranjan/ai_ids_project/data/processed/processed_data.csv")
LABEL_CLASSES_PATH = Path("/home/niranjan/ai_ids_project/data/processed/label_classes.npy")
SAVE_DIR = Path("/home/niranjan/ai_ids_project/models/saved")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TIMESTEPS = 7
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 0.001
EPOCHS = 5
BATCH_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Total rows: {len(df):,}, Columns: {df.shape[1]}")

feature_cols = [c for c in df.columns if c != "Label"]
X_full = df[feature_cols].values.astype(np.float32)
y_full = df["Label"].values.astype(np.int64)

label_classes = np.load(LABEL_CLASSES_PATH, allow_pickle=True)
num_classes = len(label_classes)
print(f"  Classes: {num_classes}")

# 20% stratified sample
print("\nSampling 20% of data (stratified)...")
X, _, y, _ = train_test_split(
    X_full, y_full, test_size=0.80, stratify=y_full, random_state=42
)
print(f"  Sampled rows: {len(X):,}")

# Pad features to be divisible by TIMESTEPS
n_features = X.shape[1]  # 77
remainder = n_features % TIMESTEPS
if remainder != 0:
    pad = TIMESTEPS - remainder
    X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
    print(f"  Padded features: {n_features} → {X.shape[1]} (pad={pad})")

input_size = X.shape[1] // TIMESTEPS  # 11
print(f"  Sequence shape per sample: ({TIMESTEPS}, {input_size})")

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# Reshape to (samples, timesteps, input_size)
X_train = X_train.reshape(-1, TIMESTEPS, input_size)
X_test  = X_test.reshape(-1, TIMESTEPS, input_size)

# Build DataLoaders
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last timestep

model = LSTMClassifier(input_size, HIDDEN_SIZE, NUM_LAYERS, num_classes, DROPOUT).to(device)
print(f"\nModel:\n{model}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

# Training loop
print("\nTraining...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * yb.size(0)

    train_acc = evaluate(train_loader)
    test_acc  = evaluate(test_loader)
    avg_loss  = running_loss / len(train_ds)
    print(f"  Epoch {epoch}/{EPOCHS} — loss: {avg_loss:.4f} | train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")

# Final test accuracy
final_acc = evaluate(test_loader)
print(f"\nFinal Test Accuracy: {final_acc:.4f}")

# Save
model_path = SAVE_DIR / "lstm_model.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "input_size": input_size,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
    "num_classes": num_classes,
    "timesteps": TIMESTEPS,
    "n_features_padded": X.shape[1],
}, model_path)
print(f"Model saved to: {model_path}")
print("\nLSTM Training Complete")
