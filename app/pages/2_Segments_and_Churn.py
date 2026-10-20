from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Segments & Churn · RetailPulse", page_icon="👥", layout="wide")

DATA = Path(__file__).resolve().parent.parent.parent / "dashboard_data"

ACTIONS = {
    "Champions": "Reward with early access and loyalty perks; protect this group",
    "Loyal Customers": "Upsell and cross-sell; offer a loyalty program",
    "Potential Loyalists": "Recent and growing; encourage a repeat purchase with bundles",
    "Recent Low-Spend": "Welcome series and offers that raise basket size",
    "At-Risk": "Send a win-back offer before they lapse completely",
    "Lost / Hibernating": "Low-cost reactivation campaign; do not over-invest",
}


@st.cache_data
def load():
    seg = pd.read_csv(DATA / "rfm_segments.csv")
    churn = pd.read_csv(DATA / "churn_scores.csv")
    return seg, churn


seg, churn = load()

st.title("👥 Customer Segments & Churn")
tab1, tab2 = st.tabs(["Segments (RFM + K-Means)", "Churn risk (XGBoost + SHAP)"])

# ---------------- Segments ----------------
with tab1:
    st.caption("Customers grouped by Recency, Frequency and Monetary value (log-scaled, K-Means, k = 6). "
               "Snapshot date: 2011-12-10.")

    profile = seg.groupby("Segment").agg(
        Customers=("Customer ID", "count"),
        Recency_days=("Recency", "median"),
        Frequency=("Frequency", "median"),
        Monetary=("Monetary", "median"),
        Total_revenue=("Monetary", "sum"),
    )
    profile["Share of customers %"] = (profile["Customers"] / profile["Customers"].sum() * 100).round(1)
    profile["Share of revenue %"] = (profile["Total_revenue"] / profile["Total_revenue"].sum() * 100).round(1)
    profile = profile.sort_values("Monetary", ascending=False)

    top = profile.iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Customers segmented", f"{int(profile['Customers'].sum()):,}")
    k2.metric(f"{profile.index[0]}: share of customers", f"{top['Share of customers %']}%")
    k3.metric(f"{profile.index[0]}: share of revenue", f"{top['Share of revenue %']}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Customers per segment")
        st.bar_chart(profile["Customers"])
    with right:
        st.subheader("Revenue share per segment (%)")
        st.bar_chart(profile["Share of revenue %"])

    st.subheader("Segment profile (median values)")
    show = profile[["Customers", "Recency_days", "Frequency", "Monetary",
                    "Share of customers %", "Share of revenue %"]].round(1)
    show["Suggested action"] = [ACTIONS.get(s, "") for s in show.index]
    st.dataframe(show)

    choice = st.selectbox("Download customers in a segment", ["All"] + sorted(seg["Segment"].unique()))
    sub = seg if choice == "All" else seg[seg["Segment"] == choice]
    st.download_button(f"Download {choice} customers (CSV)", sub.to_csv(index=False).encode(),
                       "retailpulse_segments.csv", "text/csv")

# ---------------- Churn ----------------
with tab2:
    st.caption("Churn = no purchase in the 90 days after 2011-09-09, among customers active in the "
               "prior 180 days. Features use only data before the cutoff.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CV AUC (training)", "0.780")
    m2.metric("Test AUC (target 0.88, not met)", "0.714")
    m3.metric("Precision@top 20% (target 0.75, not met)", "0.658")
    m4.metric("Base churn rate", "38.5%")
    st.caption("Baselines on the test set: logistic regression AUC 0.706, recency-only AUC 0.616. "
               "Precision@top 20% is about 1.7 times the base churn rate.")

    st.subheader("At-risk customers")
    top_pct = st.slider("Show the top N% highest-risk customers", 5, 50, 20, step=5)
    n = int(len(churn) * top_pct / 100)
    risky = churn.sort_values("ChurnProb", ascending=False).head(n)
    c1, c2 = st.columns(2)
    c1.metric("Customers scored", f"{len(churn):,}")
    c2.metric(f"Average churn probability (top {top_pct}%)", f"{risky['ChurnProb'].mean():.2f}")

    cols = ["Customer ID", "ChurnProb", "Recency", "Frequency", "Monetary", "OverdueRatio", "Orders_last90d"]
    st.dataframe(risky[cols].round(2), hide_index=True)
    st.download_button("Download at-risk list (CSV)", risky[cols].to_csv(index=False).encode(),
                       "retailpulse_at_risk.csv", "text/csv")

    st.subheader("Distribution of churn probability")
    hist = pd.cut(churn["ChurnProb"], bins=np.linspace(0, 1, 11)).value_counts().sort_index()
    hist.index = [f"{i.left:.1f}-{i.right:.1f}" for i in hist.index]
    st.bar_chart(hist)

    st.subheader("What drives churn (SHAP)")
    shap_img = DATA / "15_shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img))
        st.caption("High spend, many different products and recent orders lower churn risk. "
                   "Long gaps between orders, long silence and being overdue raise it.")
    imp_img = DATA / "16_feature_importance.png"
    if imp_img.exists():
        st.image(str(imp_img))
    st.caption("Scores are for all customers in the sample, including those used for training, "
               "so use them as a ranked at-risk list. Reported AUC comes from the held-out test set.")