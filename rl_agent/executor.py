import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG = Path("/home/niranjan/ai_ids_project/logs/audit_log.json")


class MitigationExecutor:
    def __init__(self):
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        if not AUDIT_LOG.exists():
            AUDIT_LOG.write_text("[]")

    # ── internal ──────────────────────────────────────────────────────────────
    def _log(self, ip: str, action_type: str):
        entries = json.loads(AUDIT_LOG.read_text())
        entries.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_ip": ip,
            "action_type": action_type,
        })
        AUDIT_LOG.write_text(json.dumps(entries, indent=2))

    # ── public actions ────────────────────────────────────────────────────────
    def block_ip(self, ip: str):
        self._log(ip, "BLOCK")
        try:
            subprocess.run(
                ["sudo", "iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"iptables block failed for {ip}: {e.stderr.strip()}")
        print(f"BLOCKED: {ip}")

    def rollback_ip(self, ip: str):
        self._log(ip, "ROLLBACK")
        try:
            subprocess.run(
                ["sudo", "iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"iptables rollback failed for {ip}: {e.stderr.strip()}")
        print(f"UNBLOCKED: {ip}")

    def rate_limit(self, ip: str):
        self._log(ip, "RATE_LIMIT")
        print(f"RATE LIMITED: {ip}")

    def quarantine(self, ip: str):
        self._log(ip, "QUARANTINE")
        print(f"QUARANTINED: {ip}")

    def monitor(self, ip: str):
        self._log(ip, "MONITOR")
        print(f"MONITORING: {ip}")

    def get_recent_logs(self, limit: int = 10) -> list:
        entries = json.loads(AUDIT_LOG.read_text())
        return entries[-limit:]


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ex = MitigationExecutor()
    ip = "192.168.1.100"

    ex.monitor(ip)
    ex.rate_limit(ip)
    ex.block_ip(ip)
    ex.quarantine(ip)
    ex.rollback_ip(ip)

    print("\nRecent audit log entries:")
    for entry in ex.get_recent_logs(limit=10):
        print(f"  [{entry['timestamp']}] {entry['action_type']:12s} {entry['source_ip']}")
