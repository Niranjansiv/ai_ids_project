"""
Simulates 3 unknown attacks not present in the training dataset
and sends them to the AI-IDS API to observe detection behaviour.

Usage:
    /home/niranjan/ai_ids_project/venv/bin/python3 unknown_attack.py
"""

import sys
import time
import json
import numpy as np
import requests
from datetime import datetime

sys.path.insert(0, "/home/niranjan/ai_ids_project")
from baseline.baseline_manager import BaselineChecker

DETECT_URL   = "http://localhost:8000/detect"
MITIGATE_URL = "http://localhost:8000/mitigate"
ATTACK_FEED  = "/tmp/attack_feed.jsonl"

try:
    baseline_checker = BaselineChecker()
except FileNotFoundError:
    baseline_checker = None
SLEEP_SEC    = 2
SAMPLES      = 5


def make_vector(overrides: dict, base: float = 0.1) -> list[float]:
    """Build a 77-element feature vector with a base value, applying overrides by index."""
    vec = [base] * 77
    for idx, val in overrides.items():
        vec[idx] = val
    return vec


ATTACKS = [
    {
        "name": "Slow Read Attack",
        "base": 0.1,
        "overrides": {
            0:  0.95,   # flow_duration
            4:  0.01,   # fwd_packets_per_sec
            10: 0.02,   # packet_length_mean
        },
    },
    {
        "name": "DNS Amplification",
        "base": 0.1,
        "overrides": {
            3: 0.0,     # protocol
            8: 0.99,    # bwd_packet_length_max
            2: 0.95,    # total_bwd_packets
            6: 0.99,    # dst_port
        },
    },
    {
        "name": "HTTP Flood",
        "base": 0.3,
        "overrides": {
            4:  0.99,   # fwd_packets_per_sec
            10: 0.45,   # packet_length_mean
            3:  1.0,    # protocol
        },
    },
]


def send_detect(feature_vector: list[float]) -> dict | None:
    try:
        resp = requests.post(
            DETECT_URL,
            json={"feature_vector": feature_vector},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [ERROR] /detect failed: {e}")
        return None


def send_mitigate(threat_type: str, confidence: float,
                  source_ip: str, is_anomaly: bool) -> str:
    try:
        resp = requests.post(
            MITIGATE_URL,
            json={
                "threat_type": threat_type,
                "confidence":  confidence,
                "source_ip":   source_ip,
                "is_anomaly":  is_anomaly,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("action_taken", "Unknown")
    except Exception as e:
        print(f"  [ERROR] /mitigate failed: {e}")
        return "Error"


def run_attack(attack: dict):
    name     = attack["name"]
    vec      = make_vector(attack["overrides"], base=attack["base"])
    src_ip   = "10.0.0.99"

    print(f"\n{'─' * 68}")
    print(f"  Simulating: {name}  ({SAMPLES} samples, {SLEEP_SEC}s apart)")
    print(f"{'─' * 68}")

    for i in range(1, SAMPLES + 1):
        result = send_detect(vec)
        if result is None:
            time.sleep(SLEEP_SEC)
            continue

        threat_type = result.get("threat_type", "Unknown")
        confidence  = result.get("confidence", 0.0)
        is_anomaly  = result.get("is_anomaly", False)

        action = send_mitigate(threat_type, confidence, src_ip, is_anomaly)

        # Baseline check
        if baseline_checker is not None:
            bl = baseline_checker.check(np.array(vec, dtype=np.float32))
            baseline_deviation = round(bl["deviation_score"] * 100, 1)
            baseline_status    = "ANOMALY" if bl["is_anomaly"] else "normal"
        else:
            baseline_deviation = 0.0
            baseline_status    = "n/a"

        ts = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{ts}] [UNKNOWN ATTACK] Type: {name:<22} | "
            f"DETECTED AS: {threat_type:<22} | "
            f"Confidence: {confidence:.2f} | "
            f"BASELINE: {baseline_status} ({baseline_deviation}%) | "
            f"Action: {action}"
        )

        with open(ATTACK_FEED, "a") as f:
            f.write(json.dumps({
                "timestamp":          ts,
                "attack_name":        name,
                "threat_type":        threat_type,
                "confidence":         round(confidence, 4),
                "is_anomaly":         is_anomaly,
                "action":             action,
                "baseline_deviation": baseline_deviation,
                "baseline_status":    baseline_status,
                "attack_source":      "unknown",
            }) + "\n")

        if i < SAMPLES:
            time.sleep(SLEEP_SEC)


def main():
    print("=" * 68)
    print("  AI-IDS — Unknown Attack Simulator")
    print(f"  API: {DETECT_URL}")
    print(f"  Attacks: {len(ATTACKS)}  |  Samples each: {SAMPLES}")
    print("=" * 68)

    for attack in ATTACKS:
        run_attack(attack)

    print(f"\n{'=' * 68}")
    print("  Simulation complete.")
    print("=" * 68)


if __name__ == "__main__":
    main()
