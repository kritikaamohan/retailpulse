from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Demand Forecast · RetailPulse", page_icon="📈", layout="wide")

DATA = Path(__file__).resolve().parent.parent.parent / "dashboard_data"


@st.cache_data
def load():
    daily = pd.read_csv(DATA / "daily_prepared.csv", index_col=0, parse_dates=True).iloc[:, 0]
    fc = pd.read_csv(DATA / "final_forecast.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return daily, fc


def mape_nonzero(actual, pred):
    m = actual > 0
    return (abs(actual[m] - pred[m]) / actual[m]).mean() * 100


def wape(actual, pred):
    return abs(actual - pred).sum() / actual.sum() * 100


daily, fc = load()
test = daily.reindex(fc.index)          # actual revenue on the forecast days
train = daily[daily.index < fc.index.min()]

st.title("📈 Demand Forecast")
st.caption("Daily revenue forecast: Prophet plus a Saturday-closure rule. "
           "The model was trained up to 2011-11-08 and tested on the following 30 days.")

# Controls
c1, c2 = st.columns(2)
horizon = c1.slider("Forecast horizon (days)", 7, len(fc), len(fc))
lift = c2.slider("Promo lift (%) - what-if scenario", 0, 50, 0,
                 help="Scales forecast revenue on trading days. This is a scenario, not a model estimate.")

f = fc.iloc[:horizon]
a = test.iloc[:horizon]
scenario = f.where(f == 0, f * (1 + lift / 100))   # closed days stay at zero

# Metrics on the base forecast (no promo)
k1, k2, k3, k4 = st.columns(4)
k1.metric("MAPE (trading days)", f"{mape_nonzero(a, f):.1f}%")
k2.metric("WAPE (all days)", f"{wape(a, f):.1f}%")
k3.metric("Forecast revenue", f"£{f.sum():,.0f}")
k4.metric("Scenario revenue", f"£{scenario.sum():,.0f}", f"{scenario.sum() - f.sum():+,.0f}" if lift > 0 else None)
st.caption("MAPE and WAPE compare the base forecast with actual revenue. "
           "MAPE skips zero-revenue days (Saturdays) because it divides by the actual value.")

# Chart
chart = pd.concat([train.iloc[-60:].rename("History"),
                   a.rename("Actual"),
                   f.rename("Forecast")], axis=1)
if lift > 0:
    chart["Scenario (promo)"] = scenario
st.subheader("Forecast vs actual")
st.line_chart(chart)

# Model comparison
st.subheader("Model comparison (30-day test window)")
results = pd.DataFrame({
    "Model": ["Seasonal naive", "Prophet baseline", "LSTM", "Ensemble (50/50)",
              "Ensemble + Saturday rule", "Prophet + Saturday rule (used here)"],
    "MAPE %": [29.37, 20.83, 33.32, 21.40, 21.40, 20.83],
    "WAPE %": [25.53, 26.06, 36.31, 27.48, 24.22, 19.80],
})
st.dataframe(results, hide_index=True)
st.caption("Results come from the training notebooks and are logged in MLflow. "
           "The PDF target is MAPE <= 12%; the best model here is above it.")

# Table and export
table = pd.DataFrame({"Forecast": f.round(0), "Actual": a.round(0), "Scenario": scenario.round(0)})
st.subheader("Forecast table")
st.dataframe(table)
st.download_button("Download forecast (CSV)", table.to_csv().encode(), "retailpulse_forecast.csv", "text/csv")