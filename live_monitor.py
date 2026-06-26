"""
Live packet monitor — sniffs real traffic on wlp130s0, sends to AI-IDS API.

Usage (requires root):
    sudo /home/niranjan/ai_ids_project/venv/bin/python3 live_monitor.py
"""

import sys
import signal
import threading
import time
import json
import numpy as np
import requests
from datetime import datetime
from pathlib import Path

from scapy.all import sniff, IP, TCP, UDP, ICMP
from baseline.baseline_manager import BaselineChecker

# ── Config ────────────────────────────────────────────────────────────────────
SNIFF_IFACE          = "wlp130s0"
DETECT_URL           = "http://localhost:8000/detect"
MITIGATE_URL         = "http://localhost:8000/mitigate"
CONFIDENCE_THRESHOLD = 0.85
ATTACK_FEED          = Path("/tmp/attack_feed.jsonl")

SESSION    = requests.Session()
stop_event = threading.Event()

packet_count = 0
alert_count  = 0

# Load baseline checker (optional — monitor still works if baseline missing)
try:
    baseline_checker = BaselineChecker()
except FileNotFoundError:
    baseline_checker = None
    print("[WARNING] Baseline not found — run baseline/baseline_manager.py to enable baseline checks.")


# ── API helpers ───────────────────────────────────────────────────────────────
def call_detect(features: np.ndarray) -> dict | None:
    try:
        resp = SESSION.post(
            DETECT_URL,
            json={"feature_vector": features.tolist()},
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def call_mitigate(threat_type: str, confidence: float,
                  source_ip: str, is_anomaly: bool) -> str:
    try:
        resp = SESSION.post(
            MITIGATE_URL,
            json={
                "threat_type": threat_type,
                "confidence":  confidence,
                "source_ip":   source_ip,
                "is_anomaly":  is_anomaly,
            },
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json().get("action_taken", "Unknown")
    except Exception:
        return "Error"


# ── Feature extraction ────────────────────────────────────────────────────────
PROTOCOL_MAP = {1: 0.01, 6: 0.06, 17: 0.17}

def extract_features(pkt) -> tuple[np.ndarray, str, str]:
    features = np.zeros(77, dtype=np.float32)

    if not pkt.haslayer(IP):
        return features, "0.0.0.0", "0.0.0.0"

    ip     = pkt[IP]
    src_ip = ip.src
    dst_ip = ip.dst

    for i, octet in enumerate(src_ip.split(".")):
        features[i] = int(octet) / 255.0
    for i, octet in enumerate(dst_ip.split(".")):
        features[4 + i] = int(octet) / 255.0

    features[8]  = PROTOCOL_MAP.get(ip.proto, ip.proto / 255.0)
    features[9]  = min(len(pkt) / 65535.0, 1.0)
    features[10] = ip.ihl / 15.0
    features[11] = ip.ttl / 255.0

    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        features[12] = min(tcp.sport / 65535.0, 1.0)
        features[13] = min(tcp.dport / 65535.0, 1.0)
        features[14] = int(tcp.flags) / 63.0
        features[15] = tcp.window / 65535.0
        features[16] = tcp.dataofs / 15.0
    elif pkt.haslayer(UDP):
        udp = pkt[UDP]
        features[12] = min(udp.sport / 65535.0, 1.0)
        features[13] = min(udp.dport / 65535.0, 1.0)
        features[17] = min(udp.len / 65535.0, 1.0)
    elif pkt.haslayer(ICMP):
        icmp = pkt[ICMP]
        features[18] = icmp.type / 255.0
        features[19] = icmp.code / 255.0

    payload_len  = len(bytes(pkt.payload)) if pkt.payload else 0
    features[20] = min(payload_len / 65535.0, 1.0)

    return features, src_ip, dst_ip


# ── Packet handler ────────────────────────────────────────────────────────────
def handle_packet(pkt):
    global packet_count, alert_count

    if stop_event.is_set() or not pkt.haslayer(IP):
        return

    features, src_ip, dst_ip = extract_features(pkt)
    result = call_detect(features)
    if result is None:
        return

    packet_count += 1

    threat_type  = result.get("threat_type", "Unknown")
    confidence   = result.get("confidence", 0.0)
    is_anomaly   = result.get("is_anomaly", False)
    action_taken = "—"

    if confidence > CONFIDENCE_THRESHOLD:
        alert_count += 1
        action_taken = call_mitigate(threat_type, confidence, src_ip, is_anomaly)

    # Baseline check
    if baseline_checker is not None:
        bl = baseline_checker.check(features)
        dev_pct      = bl["deviation_score"] * 100
        baseline_str = f"ANOMALY ({dev_pct:.0f}%)" if bl["is_anomaly"] else f"normal  ({dev_pct:.0f}%)"
    else:
        baseline_str = "n/a"

    ts = datetime.now().strftime("%H:%M:%S")
    print(
        f"[{ts}] {src_ip:<15} → {dst_ip:<15} | "
        f"THREAT: {threat_type:<22} | "
        f"Conf: {confidence:.2f} | "
        f"BASELINE: {baseline_str} | Action: {action_taken}"
    )


# ── Thread 2: Attack feed tailer ─────────────────────────────────────────────
def feed_tailer_thread():
    """Tail /tmp/attack_feed.jsonl and print new lines as they arrive."""
    # Seek to end so we only show lines written after startup
    offset = ATTACK_FEED.stat().st_size if ATTACK_FEED.exists() else 0

    while not stop_event.is_set():
        if ATTACK_FEED.exists():
            with open(ATTACK_FEED, "r") as f:
                f.seek(offset)
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry         = json.loads(raw)
                        ts            = entry.get("timestamp", datetime.now().strftime("%H:%M:%S"))
                        attack_source = entry.get("attack_source", "known")
                        threat_type   = entry.get("threat_type", "Unknown")
                        confidence    = entry.get("confidence", 0.0)
                        action        = entry.get("action", "—")
                        deviation     = entry.get("baseline_deviation", 0.0)
                        bl_status     = entry.get("baseline_status", "n/a")
                        baseline_tag  = f"{bl_status} ({deviation}%)"

                        if attack_source == "unknown":
                            attack_name = entry.get("attack_name", "Unknown")
                            print(
                                f"[{ts}] [⚠️  UNKNOWN ATTACK] "
                                f"Type: {attack_name:<22} | "
                                f"DETECTED AS: {threat_type:<22} | "
                                f"Conf: {confidence:.2f} | "
                                f"BASELINE: {baseline_tag} | "
                                f"Action: {action}"
                            )
                        else:
                            source_ip = entry.get("source_ip", "?")
                            print(
                                f"[{ts}] [🔴 KNOWN ATTACK] "
                                f"Type: {threat_type:<22} | "
                                f"Conf: {confidence:.2f} | "
                                f"BASELINE: {baseline_tag} | "
                                f"Action: {action:<10} | "
                                f"IP: {source_ip}"
                            )
                    except json.JSONDecodeError:
                        pass
                offset = f.tell()

        stop_event.wait(0.5)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  AI-IDS Live Monitor")
    print(f"  Interface : {SNIFF_IFACE}")
    print(f"  API       : {DETECT_URL}")
    print(f"  Threshold : confidence > {CONFIDENCE_THRESHOLD}")
    print("  Press Ctrl+C to stop")
    print("=" * 72 + "\n")

    def on_sigint(sig, frame):
        print("\n\nStopping ...")
        stop_event.set()
        print(f"Packets processed : {packet_count}")
        print(f"Alerts triggered  : {alert_count}")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    t2 = threading.Thread(target=feed_tailer_thread, daemon=True, name="feed-tailer")
    t2.start()

    sniff(
        iface=SNIFF_IFACE,
        filter="ip",
        prn=handle_packet,
        store=False,
        stop_filter=lambda _: stop_event.is_set(),
    )


if __name__ == "__main__":
    main()
