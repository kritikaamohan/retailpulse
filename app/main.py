from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="RetailPulse", page_icon="📊", layout="wide")

DATA = Path(__file__).resolve().parent.parent / "dashboard_data"


@st.cache_data
def load():
    daily = pd.read_csv(DATA / "daily_country.csv", parse_dates=["date"])
    top = pd.read_csv(DATA / "top_products.csv")
    static = pd.read_csv(DATA / "static_kpis.csv").iloc[0]
    return daily, top, static


daily, top, static = load()

st.title("📊 RetailPulse")
st.caption("AI-Powered Customer Analytics & Demand Forecasting Platform · "
           "Data: Online Retail II (UCI), UK online retailer, 2009-12 to 2011-12 · Currency: £")

# Sidebar filters
st.sidebar.header("Filters")
dmin, dmax = daily["date"].min().date(), daily["date"].max().date()
picked = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (dmin, dmax))

countries = sorted(daily["Country"].unique())
chosen = st.sidebar.multiselect("Country (empty = all)", countries)

mask = (daily["date"].dt.date >= start) & (daily["date"].dt.date <= end)
if chosen:
    mask &= daily["Country"].isin(chosen)
view = daily[mask]

# KPI cards
c1, c2, c3, c4 = st.columns(4)
revenue, orders, units = view["Revenue"].sum(), view["Orders"].sum(), view["Units"].sum()
c1.metric("Revenue", f"£{revenue:,.0f}")
c2.metric("Orders", f"{orders:,}")
c3.metric("Avg order value", f"£{revenue / orders:,.0f}" if orders else "n/a")
c4.metric("Units sold", f"{units:,}")
st.caption(f"Whole period: {int(static['customers']):,} customers and "
           f"{int(static['products']):,} products (these two do not change with the filters).")

@st.cache_data
def load_alerts():
    inv = pd.read_csv(DATA / "inventory_recommendations.csv")
    churn = pd.read_csv(DATA / "churn_scores.csv")
    drift = pd.read_csv(DATA / "drift_summary.csv")
    return inv, churn, drift


inv, churn, drift = load_alerts()

st.subheader("🔔 Alerts")
a1, a2, a3 = st.columns(3)

at_risk = int((churn["ChurnProb"] >= 0.70).sum())
with a1:
    st.warning(f"**{at_risk:,} customers** had a churn probability of 70% or more (scored as of 9 Sep 2011).")
    st.caption("See Segments & Churn for the list.")

inv = inv.assign(StockPct=inv["CurrentStock"] / inv["ReorderPoint"] * 100)
low = inv[inv["Status"] == "Reorder"].sort_values("StockPct")
with a2:
    if len(low):
        worst = low.iloc[0]
        st.error(f"**{len(low)} products** are at or below their reorder point. "
                 f"Lowest: {worst['StockCode']} at {worst['StockPct']:.0f}% of its reorder point.")
    else:
        st.success("No products are below their reorder point.")
    st.caption("Stock levels are simulated.")

real = drift[drift["batch"] == "real_test"]
with a3:
    if len(real) and bool(real.iloc[0]["alert"]):
        st.error(f"Data drift alert: {real.iloc[0]['share'] * 100:.0f}% of features drifted.")
    else:
        share = real.iloc[0]["share"] * 100 if len(real) else 0
        st.success(f"No data drift alert ({share:.0f}% of features drifted, threshold 30%).")
    st.caption("See Monitoring for the Evidently reports.")
st.divider()

# Daily revenue
st.subheader("Daily revenue")
series = view.groupby("date")["Revenue"].sum().to_frame("Revenue")
series["7-day average"] = series["Revenue"].rolling(7).mean()
st.line_chart(series)
st.caption("Zero-revenue days are mostly Saturdays, when the shop does not trade.")

left, right = st.columns(2)
with left:
    st.subheader("Monthly revenue")
    monthly = view.set_index("date")["Revenue"].resample("MS").sum()
    st.bar_chart(monthly)
with right:
    st.subheader("Top 10 countries by revenue")
    by_country = view.groupby("Country")["Revenue"].sum().nlargest(10)
    st.bar_chart(by_country)

st.subheader("Top 15 products by revenue (whole period)")
st.dataframe(top.assign(Revenue=top["Revenue"].round(0)), hide_index=True)

st.download_button("Download filtered daily data (CSV)",
                   view.to_csv(index=False).encode(), "retailpulse_filtered.csv", "text/csv")
