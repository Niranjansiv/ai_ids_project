import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── Config ────────────────────────────────────────────────────────────────────
AUDIT_LOG   = Path("/home/niranjan/ai_ids_project/logs/audit_log.json")
MODELS_DIR  = Path("/home/niranjan/ai_ids_project/models/saved")
MODEL_FILES = ["autoencoder.h5", "rf_model.pkl", "lstm_model.pt", "isolation_forest.pkl"]

BLUE_PRIMARY   = "#1565C0"
BLUE_LIGHT     = "#42A5F5"
BLUE_DARK      = "#0D47A1"
BG_CARD        = "#E3F2FD"

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh every 5 seconds
st_autorefresh(interval=5_000, key="autorefresh")

# ── Custom CSS (blue theme) ───────────────────────────────────────────────────
st.markdown(f"""
<style>
    .main {{background-color: #F0F7FF;}}
    .stApp {{background-color: #F0F7FF;}}
    .metric-card {{
        background-color: {BG_CARD};
        border-left: 5px solid {BLUE_PRIMARY};
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }}
    .metric-title {{color: {BLUE_DARK}; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;}}
    .metric-value {{color: {BLUE_PRIMARY}; font-size: 36px; font-weight: 700; line-height: 1.2;}}
    .section-header {{
        color: {BLUE_DARK};
        font-size: 18px;
        font-weight: 600;
        border-bottom: 2px solid {BLUE_LIGHT};
        padding-bottom: 4px;
        margin: 16px 0 12px 0;
    }}
    [data-testid="stSidebar"] {{background-color: {BLUE_DARK}; color: white;}}
    [data-testid="stSidebar"] * {{color: white !important;}}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
_EMPTY_DF = pd.DataFrame(columns=[
    "id", "timestamp", "source_ip", "threat_type",
    "confidence", "action_type", "action_taken", "explanation", "is_anomaly",
])

def load_logs() -> pd.DataFrame:
    # Reset file if missing, empty, or oversized (> 5 MB → likely corrupted)
    if not AUDIT_LOG.exists() or AUDIT_LOG.stat().st_size == 0:
        return _EMPTY_DF.copy()
    if AUDIT_LOG.stat().st_size > 5 * 1024 * 1024:
        AUDIT_LOG.write_text("[]")
        return _EMPTY_DF.copy()

    try:
        entries = json.loads(AUDIT_LOG.read_text())
    except json.JSONDecodeError:
        AUDIT_LOG.write_text("[]")
        return _EMPTY_DF.copy()

    if not entries:
        return _EMPTY_DF.copy()

    df = pd.DataFrame(entries)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Ensure both action columns exist so metrics and table work regardless of source
    if "action_type" not in df.columns:
        df["action_type"] = df.get("action_taken", "Monitor")
    if "action_taken" not in df.columns:
        df["action_taken"] = df["action_type"]

    if "threat_type" not in df.columns:
        df["threat_type"] = "Unknown"
    if "confidence" not in df.columns:
        df["confidence"] = 0.0
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = False

    return df


def today_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    today = datetime.now(timezone.utc).date()
    return df[df["timestamp"].dt.date == today]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ AI-IDS System")
    st.markdown("---")
    st.markdown("### Model Status")
    for fname in MODEL_FILES:
        exists = (MODELS_DIR / fname).exists()
        icon   = "🟢" if exists else "🔴"
        label  = fname.replace("_", " ").replace(".pkl", "").replace(".h5", "").replace(".pt", "").replace(".npy", "")
        st.markdown(f"{icon} **{label}**")
    st.markdown("---")
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()
    st.markdown("---")
    st.caption("Auto-refreshes every 5 seconds")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown(f"<h1 style='color:{BLUE_DARK};'>🛡️ AI Intrusion Detection System</h1>", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_logs()

# Today's slice — used only for the line chart
df_today = today_df(df)

# ── Metrics (cumulative all-time) ─────────────────────────────────────────────
if not df.empty and "threat_type" in df.columns:
    # total: rows that have a non-null threat_type
    total_alerts = int(df["threat_type"].notna().sum())

    # blocked: action_type == "BLOCK"  OR  action_taken == "BlockIP"
    blocked_mask = pd.Series(False, index=df.index)
    if "action_type"  in df.columns:
        blocked_mask |= df["action_type"].str.upper().str.strip().eq("BLOCK")
    if "action_taken" in df.columns:
        blocked_mask |= df["action_taken"].str.strip().eq("BlockIP")
    threats_blocked = int(blocked_mask.sum())

    # benign: threat_type == "Benign"
    benign_count = int(df["threat_type"].eq("Benign").sum())

    # anomalies: threat_type is set, non-empty, and not "Benign"
    anomalies = int(
        df["threat_type"]
        .loc[df["threat_type"].notna() & df["threat_type"].str.strip().ne("") & df["threat_type"].ne("Benign")]
        .count()
    )
else:
    total_alerts = threats_blocked = benign_count = anomalies = 0

# ── Metric cards ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

def metric_card(container, title, value, color=BLUE_PRIMARY):
    container.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)

metric_card(col1, "Total Alerts",      total_alerts)
metric_card(col2, "Threats Blocked",   threats_blocked, "#B71C1C")
metric_card(col3, "Benign Traffic",    benign_count,    "#1B5E20")
metric_card(col4, "Anomalies Detected", anomalies,      "#E65100")

# ── Charts ────────────────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns([3, 2])

with chart_col1:
    st.markdown("<div class='section-header'>Threat Count Over Time — Today (by Hour)</div>", unsafe_allow_html=True)
    if not df_today.empty and "timestamp" in df_today.columns:
        df_plot = df_today.copy()
        df_plot["hour"] = df_plot["timestamp"].dt.floor("h")
        hourly = (
            df_plot.groupby(["hour", "threat_type"])
            .size()
            .reset_index(name="count")
        )
        fig_line = px.line(
            hourly,
            x="hour", y="count", color="threat_type",
            labels={"hour": "Time", "count": "Events", "threat_type": "Threat Type"},
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_line.update_layout(
            plot_bgcolor="#F0F7FF",
            paper_bgcolor="#F0F7FF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(color=BLUE_DARK),
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No data today yet — waiting for detections.")

with chart_col2:
    st.markdown("<div class='section-header'>Attack Type Distribution (All Time)</div>", unsafe_allow_html=True)
    if not df.empty and "threat_type" in df.columns:
        dist = df["threat_type"].value_counts().reset_index()
        dist.columns = ["threat_type", "count"]
        fig_pie = px.pie(
            dist,
            names="threat_type", values="count",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.35,
        )
        fig_pie.update_layout(
            paper_bgcolor="#F0F7FF",
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="v"),
            font=dict(color=BLUE_DARK),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No data yet.")

# ── Recent threats table (last 10 from all-time data) ─────────────────────────
st.markdown("<div class='section-header'>Last 10 Detected Threats</div>", unsafe_allow_html=True)

if not df.empty:
    action_col = "action_taken" if "action_taken" in df.columns else "action_type"
    table_cols = ["timestamp", "source_ip", "threat_type", "confidence", action_col]
    available  = [c for c in table_cols if c in df.columns]
    recent     = df.sort_values("timestamp", ascending=False).head(10)[available].copy()
    recent["timestamp"] = recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if "confidence" in recent.columns:
        recent["confidence"] = recent["confidence"].apply(lambda v: f"{float(v):.2f}" if pd.notna(v) else "—")
    st.dataframe(recent, use_container_width=True, hide_index=True)
else:
    st.info("No log entries found yet. Start sending traffic through the API to see results here.")
