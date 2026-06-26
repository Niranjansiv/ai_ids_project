"""
Baseline manager for AI-IDS.

Usage:
    # Establish baseline from dataset
    /home/niranjan/ai_ids_project/venv/bin/python3 baseline/baseline_manager.py

    # Import in other scripts
    from baseline.baseline_manager import BaselineChecker
"""

import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

DATA_PATH      = Path("/home/niranjan/ai_ids_project/data/processed/processed_data.csv")
BASELINE_PATH  = Path("/home/niranjan/ai_ids_project/models/saved/baseline_stats.json")
BENIGN_SAMPLE  = 1000
DEVIATION_STD  = 2.0      # features outside mean ± N*std count as deviating
ANOMALY_THRESH = 0.40     # >40% of features deviating → anomaly


# ── 1. Establish baseline ─────────────────────────────────────────────────────
def establish_baseline(save: bool = True) -> dict:
    """
    Load processed CSV, take the last BENIGN_SAMPLE benign rows,
    compute per-feature stats, and save to baseline_stats.json.
    Returns the stats dict.
    """
    print("Loading dataset ...")
    df = pd.read_csv(DATA_PATH)

    benign = df[df["Label"] == 0].copy()
    sample = benign.tail(BENIGN_SAMPLE)
    feature_cols = [c for c in df.columns if c != "Label"]
    X = sample[feature_cols].values.astype(np.float64)

    stats = {
        "n_samples":    int(len(X)),
        "n_features":   int(X.shape[1]),
        "feature_names": feature_cols,
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "features": {}
    }

    for idx, name in enumerate(feature_cols):
        col = X[:, idx]
        stats["features"][name] = {
            "mean":   float(np.mean(col)),
            "std":    float(np.std(col)),
            "min":    float(np.min(col)),
            "max":    float(np.max(col)),
            "p95":    float(np.percentile(col, 95)),
        }

    if save:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(stats, indent=2))
        print(f"Baseline established from {len(X)} benign samples → {BASELINE_PATH}")

    return stats


# ── 2. BaselineChecker ────────────────────────────────────────────────────────
class BaselineChecker:
    """
    Loads baseline_stats.json and checks feature vectors against it.

    checker = BaselineChecker()
    result  = checker.check(feature_vector)   # np.ndarray of length 77
    # result keys: deviation_score, deviating_features, is_anomaly
    """

    def __init__(self, path: Path = BASELINE_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Baseline not found at {path}. "
                "Run baseline_manager.py first to establish it."
            )
        raw = json.loads(path.read_text())
        self.feature_names: list[str] = raw["feature_names"]
        self.n_features: int          = raw["n_features"]

        # Pre-build arrays for fast numpy ops
        self._mean = np.array([raw["features"][n]["mean"] for n in self.feature_names], dtype=np.float64)
        self._std  = np.array([raw["features"][n]["std"]  for n in self.feature_names], dtype=np.float64)
        # Avoid division/comparison issues when std == 0
        self._std  = np.where(self._std == 0, 1e-9, self._std)

    def check(self, feature_vector: np.ndarray) -> dict:
        """
        Compare feature_vector against baseline statistics.

        Returns:
            deviation_score   : float  0-1  (fraction of deviating features)
            deviating_features: list   names of features outside mean ± 2σ
            is_anomaly        : bool   True if deviation_score > ANOMALY_THRESH
        """
        fv = np.asarray(feature_vector, dtype=np.float64)

        # Pad or trim to match baseline length
        if len(fv) < self.n_features:
            fv = np.pad(fv, (0, self.n_features - len(fv)))
        else:
            fv = fv[: self.n_features]

        z_scores   = np.abs((fv - self._mean) / self._std)
        deviating_mask = z_scores > DEVIATION_STD

        deviating_features = [
            self.feature_names[i] for i in np.where(deviating_mask)[0]
        ]
        deviation_score = float(deviating_mask.sum() / self.n_features)
        is_anomaly      = deviation_score > ANOMALY_THRESH

        return {
            "deviation_score":    round(deviation_score, 4),
            "deviating_features": deviating_features,
            "is_anomaly":         is_anomaly,
        }


# ── 3. Baseline updater ───────────────────────────────────────────────────────
def update_baseline() -> dict:
    """Re-establish baseline from the dataset and overwrite the saved file."""
    stats = establish_baseline(save=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Baseline updated at {ts}")
    return stats


# ── 4. Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Establish ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1: Establishing baseline")
    print("=" * 60)
    establish_baseline()

    checker = BaselineChecker()
    print(f"\nBaselineChecker loaded ({checker.n_features} features)")

    # ── Test 1: normal benign vector (all zeros = scaled minimum traffic) ──────
    print("\n" + "=" * 60)
    print("  STEP 2: Test with a NORMAL (zero/low) vector")
    print("=" * 60)
    normal_vec = np.zeros(77, dtype=np.float32)
    result_normal = checker.check(normal_vec)
    label = "ANOMALY" if result_normal["is_anomaly"] else "normal"
    print(f"  Deviation score : {result_normal['deviation_score']:.4f} "
          f"({result_normal['deviation_score'] * 100:.1f}%)")
    print(f"  Deviating feats : {len(result_normal['deviating_features'])}")
    print(f"  Verdict         : {label}")

    # ── Test 2: high-value random vector (simulates attack traffic) ───────────
    print("\n" + "=" * 60)
    print("  STEP 3: Test with a HIGH-VALUE (random) vector")
    print("=" * 60)
    rng = np.random.default_rng(seed=0)
    attack_vec = rng.uniform(0.5, 1.0, size=77).astype(np.float32)
    result_attack = checker.check(attack_vec)
    label = "ANOMALY" if result_attack["is_anomaly"] else "normal"
    print(f"  Deviation score : {result_attack['deviation_score']:.4f} "
          f"({result_attack['deviation_score'] * 100:.1f}%)")
    print(f"  Deviating feats : {len(result_attack['deviating_features'])}")
    if result_attack["deviating_features"]:
        print(f"  Top deviators   : {result_attack['deviating_features'][:5]}")
    print(f"  Verdict         : {label}")

    # ── Comparison summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Normal vector  → deviation {result_normal['deviation_score'] * 100:5.1f}%  "
          f"| {'normal' if not result_normal['is_anomaly'] else "ANOMALY":>8}")
    print(f"  Attack vector  → deviation {result_attack['deviation_score'] * 100:5.1f}%  "
          f"| {'normal' if not result_attack['is_anomaly'] else "ANOMALY":>8}")
    print()
