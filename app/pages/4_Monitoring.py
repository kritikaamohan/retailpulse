from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Monitoring · RetailPulse", page_icon="🛰️", layout="wide")

DATA = Path(__file__).resolve().parent.parent.parent / "dashboard_data"
THRESHOLD = 0.30


@st.cache_data
def load():
    summary = pd.read_csv(DATA / "drift_summary.csv")
    churn = pd.read_csv(DATA / "churn_scores.csv")
    return summary, churn


def psi(ref, cur, bins=10):
    """Population Stability Index of one feature (reference vs current)."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


summary, churn = load()

st.title("🛰️ Model Monitoring & Drift")
st.caption("Drift = the data the model sees in production no longer looks like the data it was trained on. "
           "Detected with Evidently; a retraining run is triggered when more than 30% of features drift.")

# ---- Drift summary ----
st.subheader("Drift check results")
cols = st.columns(len(summary))
for col, (_, r) in zip(cols, summary.iterrows()):
    label = r["batch"].replace("_", " ").title()
    col.metric(label, f"{r['share'] * 100:.0f}% of features drifted",
               "ALERT: retrain" if r["alert"] else "OK: no retraining", delta_color="off")
st.caption("Real test batch: same period as training, so little drift is expected. "
           "Simulated drift: behaviour deliberately shifted (longer silence, fewer recent orders) to test the alert. "
           "Alert threshold: 30% of features.")

# ---- Evidently report viewer ----
st.subheader("Evidently drift reports")
reports = {"Real test batch": "drift_real_test.html", "Simulated drift": "drift_simulated.html"}
available = {k: v for k, v in reports.items() if (DATA / v).exists()}
if available:
    pick = st.selectbox("Report", list(available))
    path = DATA / available[pick]
    if hasattr(st, "iframe"):                      # newer Streamlit
        st.iframe(path, height=700)
    else:                                          # older Streamlit
        components.html(path.read_text(encoding="utf-8"), height=700, scrolling=True)
else:
    st.info("Evidently HTML reports are not in dashboard_data yet.")

# ---- Interactive drift demo ----
st.subheader("Try it: simulate drift")
st.caption("A lightweight demo using the Population Stability Index (PSI), a simpler measure than the "
           "statistical tests Evidently uses. PSI above 0.2 is usually read as a significant shift.")
a, b = st.columns(2)
silence = a.slider("Customers silent for longer (Recency, +%)", 0, 100, 0, step=10)
spend = b.slider("Customers spend less (Monetary, -%)", 0, 60, 0, step=10)

ref = churn.copy()
cur = churn.copy()
cur["Recency"] = cur["Recency"] * (1 + silence / 100)
cur["Monetary"] = cur["Monetary"] * (1 - spend / 100)
cur["OverdueRatio"] = cur["OverdueRatio"] * (1 + silence / 100)

features = ["Recency", "Monetary", "OverdueRatio", "Frequency", "AvgOrderValue"]
features = [f for f in features if f in churn.columns]
if "AvgOrderValue" in cur:
    cur["AvgOrderValue"] = cur["AvgOrderValue"] * (1 - spend / 100)
scores = pd.Series({f: psi(ref[f].values, cur[f].values) for f in features}, name="PSI")
drifted = int((scores > 0.2).sum())

d1, d2 = st.columns(2)
d1.metric("Features with PSI > 0.2", f"{drifted} of {len(features)}")
d2.metric("Status", "ALERT: retrain" if drifted / len(features) > THRESHOLD else "OK")
st.bar_chart(scores)

# ---- MLOps summary ----
st.subheader("Retraining loop")
st.markdown("""
1. **Validate** the data (Great Expectations checks: no null IDs, positive price and quantity).
2. **Compare** current customer features with the reference batch (Evidently).
3. **Retrain** the churn model if the drift share is above 30%, and log the run in MLflow.

The script is `src/retrain.py`; `dags/retailpulse_dag.py` schedules it weekly in Airflow.
In a test run on real data the check reported **29%** (no retraining); with simulated drift it reported **71%** (retraining triggered).
""")
st.caption("A limitation found while building this: comparing features across different seasons or history lengths "
           "caused false alarms (50-55% drift on normal data), so the check uses only features that do not grow "
           "with history length.")