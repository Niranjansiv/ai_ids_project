import sys
import time
import random
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, "/home/niranjan/ai_ids_project")
from baseline.baseline_manager import BaselineChecker

ATTACK_FEED = "/tmp/attack_feed.jsonl"

DATA_PATH  = "/home/niranjan/ai_ids_project/data/processed/processed_data.csv"
DETECT_URL = "http://localhost:8000/detect"
MITIGATE_URL = "http://localhost:8000/mitigate"
MAX_ROWS   = 200
SLEEP_SEC  = 0.5

try:
    baseline_checker = BaselineChecker()
    print("Baseline checker loaded.")
except FileNotFoundError:
    baseline_checker = None
    print("WARNING: Baseline not found — run baseline/baseline_manager.py first.")

print("Loading data...")
df = pd.read_csv(DATA_PATH)
attacks = df[df["Label"] != 0].copy()
print(f"  Total attack rows: {len(attacks):,}")

attacks = attacks.sample(frac=1, random_state=42).reset_index(drop=True)
feature_cols = [c for c in attacks.columns if c != "Label"]

print(f"  Processing first {MAX_ROWS} rows...\n")

rng = random.Random(42)

for i, (_, row) in enumerate(attacks.iterrows()):
    if i >= MAX_ROWS:
        break

    source_ip = f"192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    feature_vector = row[feature_cols].tolist()

    # Detect
    try:
        det_resp = requests.post(DETECT_URL, json={"feature_vector": feature_vector}, timeout=10)
        det_resp.raise_for_status()
        detection = det_resp.json()
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR on /detect row {i}: {e}")
        continue

    threat_type   = detection.get("threat_type", "Unknown")
    confidence    = detection.get("confidence", 0.0)
    is_anomaly    = detection.get("is_anomaly", False)
    anomaly_score = detection.get("anomaly_score", 0.0)

    # Mitigate
    try:
        mit_resp = requests.post(MITIGATE_URL, json={
            "threat_type": threat_type,
            "confidence":  confidence,
            "source_ip":   source_ip,
            "is_anomaly":  is_anomaly,
        }, timeout=10)
        mit_resp.raise_for_status()
        action_taken = mit_resp.json().get("action_taken", "Unknown")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR on /mitigate row {i}: {e}")
        action_taken = "Error"

    # Baseline check
    if baseline_checker is not None:
        bl = baseline_checker.check(np.array(feature_vector, dtype=np.float32))
        baseline_deviation = round(bl["deviation_score"] * 100, 1)
        baseline_status    = "ANOMALY" if bl["is_anomaly"] else "normal"
    else:
        baseline_deviation = 0.0
        baseline_status    = "n/a"

    ts = datetime.now().strftime("%H:%M:%S")
    print(
        f"[{ts}] THREAT: {threat_type:<20} | "
        f"Confidence: {confidence:.2f} | "
        f"BASELINE: {baseline_status} ({baseline_deviation}%) | "
        f"Action: {action_taken:<10} | "
        f"IP: {source_ip}"
    )

    with open(ATTACK_FEED, "a") as f:
        f.write(json.dumps({
            "timestamp":          ts,
            "threat_type":        threat_type,
            "confidence":         round(confidence, 4),
            "action":             action_taken,
            "source_ip":          source_ip,
            "baseline_deviation": baseline_deviation,
            "baseline_status":    baseline_status,
            "attack_source":      "known",
        }) + "\n")

    time.sleep(SLEEP_SEC)

print(f"\nPipeline complete. Processed {min(i + 1, MAX_ROWS)} rows.")
