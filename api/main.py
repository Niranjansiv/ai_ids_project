import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from detection.detector import ThreatDetector
from rl_agent.executor import MitigationExecutor
from api.database import save_log, get_recent_logs

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="AI-IDS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ───────────────────────────────────────────────
print("Initialising models...")
detector = ThreatDetector()
executor = MitigationExecutor()
print("All models ready.\n")

# ── Schemas ───────────────────────────────────────────────────────────────────
class FeatureRequest(BaseModel):
    feature_vector: list[float]

    @field_validator("feature_vector")
    @classmethod
    def check_length(cls, v):
        if len(v) != 77:
            raise ValueError(f"feature_vector must have exactly 77 elements, got {len(v)}")
        return v


class MitigateRequest(BaseModel):
    threat_type: str
    confidence:  float
    source_ip:   str
    is_anomaly:  bool


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}


@app.post("/detect")
def detect(req: FeatureRequest):
    x = np.array(req.feature_vector, dtype=np.float32)
    result = detector.detect(x)

    save_log({
        "source_ip":    "unknown",
        "threat_type":  result["threat_type"],
        "confidence":   result["confidence"],
        "action_taken": "detect",
        "explanation":  result["source"],
        "is_anomaly":   result["is_anomaly"],
    })

    return {
        "threat_type":   result["threat_type"],
        "confidence":    result["confidence"],
        "is_anomaly":    result["is_anomaly"],
        "anomaly_score": result["anomaly_score"],
    }


@app.post("/mitigate")
def mitigate(req: MitigateRequest):
    if req.confidence > 0.9:
        executor.block_ip(req.source_ip)
        action_taken = "BlockIP"
    elif req.confidence > 0.7:
        executor.rate_limit(req.source_ip)
        action_taken = "RateLimit"
    else:
        executor.monitor(req.source_ip)
        action_taken = "Monitor"

    save_log({
        "source_ip":    req.source_ip,
        "threat_type":  req.threat_type,
        "confidence":   req.confidence,
        "action_taken": action_taken,
        "explanation":  f"Automated mitigation: confidence={req.confidence:.2f}",
        "is_anomaly":   req.is_anomaly,
    })

    return {"action_taken": action_taken, "source_ip": req.source_ip}


@app.get("/logs")
def logs():
    return {"logs": get_recent_logs(limit=100)}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
