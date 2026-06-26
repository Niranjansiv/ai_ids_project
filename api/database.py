import json
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG = Path("/home/niranjan/ai_ids_project/logs/audit_log.json")

_lock = threading.Lock()


def _load() -> list:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not AUDIT_LOG.exists():
        AUDIT_LOG.write_text("[]")
        return []
    try:
        return json.loads(AUDIT_LOG.read_text())
    except json.JSONDecodeError:
        AUDIT_LOG.write_text("[]")
        return []


def _save(entries: list):
    AUDIT_LOG.write_text(json.dumps(entries, indent=2))


def save_log(log_dict: dict) -> dict:
    """
    Append a log entry. Injects id and timestamp if not already present.
    Expected keys: source_ip, threat_type, confidence, action_taken,
                   explanation, is_anomaly
    """
    entry = {
        "id":           str(uuid.uuid4()),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "source_ip":    log_dict.get("source_ip", "unknown"),
        "threat_type":  log_dict.get("threat_type", "unknown"),
        "confidence":   log_dict.get("confidence", 0.0),
        "action_taken": log_dict.get("action_taken", "none"),
        "explanation":  log_dict.get("explanation", ""),
        "is_anomaly":   log_dict.get("is_anomaly", False),
    }
    with _lock:
        entries = _load()
        entries.append(entry)
        _save(entries)
    return entry


def get_recent_logs(limit: int = 100) -> list:
    """Return the last `limit` log entries, newest last."""
    with _lock:
        entries = _load()
    return entries[-limit:]
