from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

from app.utils.io import read_json
from app.utils.paths import ALERT_STORE_PATH


API_BASE = os.getenv("BARINGS_API_BASE_URL", "http://localhost:8000")


def fetch_json(path: str, default):
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return default


st.set_page_config(page_title="Barings Fraud Platform", layout="wide")
st.title("Barings Fraud & OpRisk Platform")
st.caption("Prototype for Barings-style rogue trading detection using public and synthetic data.")

metrics = fetch_json("/metrics", {})
timeline = fetch_json("/timeline", [])
sources = fetch_json("/sources", [])
alerts = read_json(ALERT_STORE_PATH, default=[])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Alerts", metrics.get("total_alerts", len(alerts)))
col2.metric("Critical Alerts", metrics.get("critical_alerts", 0))
col3.metric("ROC-AUC", round(metrics.get("model_metrics", {}).get("roc_auc", 0.0), 3))
col4.metric("Last Train", metrics.get("trained_at", "n/a"))

st.subheader("Alert Table")
if alerts:
    alert_frame = pd.DataFrame(alerts)
    st.dataframe(alert_frame[["alert_id", "date", "entity_id", "risk_score", "risk_band", "narrative"]], use_container_width=True)
else:
    st.info("No alerts stored yet. Run the bootstrap or call /train and /predict first.")

left, right = st.columns([1.3, 1.0])

with left:
    st.subheader("Timeline Explorer")
    if timeline:
        timeline_frame = pd.DataFrame(timeline)
        st.dataframe(timeline_frame[["date", "description", "source_id", "source_title"]], use_container_width=True)
    else:
        st.warning("Timeline data is unavailable.")

with right:
    st.subheader("Sources")
    if sources:
        source_frame = pd.DataFrame(sources)
        st.dataframe(source_frame[["id", "date", "title", "publisher", "reliability"]], use_container_width=True)
    else:
        st.warning("Source registry is unavailable.")

st.subheader("Alert Drill-Down")
if alerts:
    options = {f"{alert['alert_id']} | {alert['date']} | {alert['risk_band']}": alert for alert in alerts}
    choice = st.selectbox("Select alert", list(options))
    selected = options[choice]
    st.write(selected["narrative"])
    st.json(selected["top_features"])

