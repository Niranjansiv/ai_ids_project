import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pickle
from pathlib import Path

import torch
import torch.nn as nn


MODELS_DIR = Path("/home/niranjan/ai_ids_project/models/saved")
DATA_DIR   = Path("/home/niranjan/ai_ids_project/data/processed")


# ── LSTM architecture (must match train_lstm.py) ──────────────────────────────
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
        return self.fc(out[:, -1, :])


# ── Detector ──────────────────────────────────────────────────────────────────
class ThreatDetector:
    def __init__(self):
        print("Loading models...")

        # Label classes
        self.label_classes = np.load(DATA_DIR / "label_classes.npy", allow_pickle=True)

        # Random Forest
        with open(MODELS_DIR / "rf_model.pkl", "rb") as f:
            self.rf = pickle.load(f)
        print("  [✓] Random Forest loaded")

        # Autoencoder (TensorFlow / Keras)
        import tensorflow as tf
        self.ae = tf.keras.models.load_model(MODELS_DIR / "autoencoder.h5", compile=False)
        self.ae_threshold = float(np.load(MODELS_DIR / "threshold.npy"))
        print(f"  [✓] Autoencoder loaded  (threshold={self.ae_threshold:.6f})")

        # Isolation Forest
        with open(MODELS_DIR / "isolation_forest.pkl", "rb") as f:
            self.iso = pickle.load(f)
        print("  [✓] Isolation Forest loaded")

        # LSTM (PyTorch)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(MODELS_DIR / "lstm_model.pt", map_location=self.device, weights_only=False)
        self.lstm_timesteps      = ckpt["timesteps"]           # 7
        self.lstm_input_size     = ckpt["input_size"]          # 11
        self.lstm_n_features_pad = ckpt["n_features_padded"]   # 77
        self.lstm = LSTMClassifier(
            input_size  = ckpt["input_size"],
            hidden_size = ckpt["hidden_size"],
            num_layers  = ckpt["num_layers"],
            dropout     = ckpt["dropout"],
            num_classes = ckpt["num_classes"],
        ).to(self.device)
        self.lstm.load_state_dict(ckpt["model_state_dict"])
        self.lstm.eval()
        print(f"  [✓] LSTM loaded  (device={self.device})")
        print("All models ready.\n")

    def detect(self, feature_vector: np.ndarray) -> dict:
        """
        Parameters
        ----------
        feature_vector : np.ndarray of shape (77,)
            MinMax-scaled feature row (same preprocessing as training data).

        Returns
        -------
        dict with keys:
            threat_type   – predicted class label (str)
            confidence    – combined confidence [0, 1]
            is_anomaly    – bool (autoencoder flag)
            anomaly_score – float (isolation forest score, lower = more anomalous)
            source        – str describing how the decision was made
        """
        x = np.array(feature_vector, dtype=np.float32).flatten()
        if x.shape[0] != 77:
            raise ValueError(f"Expected 77 features, got {x.shape[0]}")

        # ── 1. Random Forest ──────────────────────────────────────────────────
        rf_proba  = self.rf.predict_proba(x.reshape(1, -1))[0]   # (15,)
        rf_class  = int(np.argmax(rf_proba))
        rf_conf   = float(rf_proba[rf_class])

        # ── 2. LSTM ───────────────────────────────────────────────────────────
        # Pad to n_features_padded if needed, then reshape to (1, T, input_size)
        x_pad = x
        if x_pad.shape[0] < self.lstm_n_features_pad:
            x_pad = np.pad(x_pad, (0, self.lstm_n_features_pad - x_pad.shape[0]))
        x_seq = torch.tensor(
            x_pad.reshape(1, self.lstm_timesteps, self.lstm_input_size),
            dtype=torch.float32,
        ).to(self.device)

        with torch.no_grad():
            logits     = self.lstm(x_seq)                          # (1, 15)
            lstm_proba = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (15,)
        lstm_class = int(np.argmax(lstm_proba))
        lstm_conf  = float(lstm_proba[lstm_class])

        # ── 3. Weighted ensemble (RF 0.6 + LSTM 0.4) ─────────────────────────
        combined_proba = 0.6 * rf_proba + 0.4 * lstm_proba
        final_class    = int(np.argmax(combined_proba))
        final_conf     = float(combined_proba[final_class])
        threat_label   = str(self.label_classes[final_class])

        # ── 4. Autoencoder anomaly detection ─────────────────────────────────
        ae_input        = x.reshape(1, -1)
        reconstruction  = self.ae.predict(ae_input, verbose=0)
        mse             = float(np.mean(np.square(ae_input - reconstruction)))
        is_anomaly      = bool(mse > self.ae_threshold)

        # ── 5. Isolation Forest anomaly score ────────────────────────────────
        # score_samples returns negative average log-likelihood; lower = more anomalous
        anomaly_score = float(self.iso.score_samples(ae_input)[0])

        # ── Source description ────────────────────────────────────────────────
        source_parts = [
            f"RF({self.label_classes[rf_class]},{rf_conf:.2f})",
            f"LSTM({self.label_classes[lstm_class]},{lstm_conf:.2f})",
        ]
        if is_anomaly:
            source_parts.append(f"AE(anomaly,mse={mse:.6f})")
        else:
            source_parts.append(f"AE(normal,mse={mse:.6f})")
        source_parts.append(f"IF(score={anomaly_score:.4f})")

        return {
            "threat_type"  : threat_label,
            "confidence"   : round(final_conf, 4),
            "is_anomaly"   : is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "source"       : " | ".join(source_parts),
        }


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = ThreatDetector()

    print("=" * 60)
    print("Test 1: random feature vector")
    rng = np.random.default_rng(0)
    test_vec = rng.random(77).astype(np.float32)
    result = detector.detect(test_vec)
    for k, v in result.items():
        print(f"  {k:<15}: {v}")

    print()
    print("Test 2: zero vector (all-benign-like)")
    zero_vec = np.zeros(77, dtype=np.float32)
    result2 = detector.detect(zero_vec)
    for k, v in result2.items():
        print(f"  {k:<15}: {v}")
