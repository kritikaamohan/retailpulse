from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Inventory · RetailPulse", page_icon="📦", layout="wide")

DATA = Path(__file__).resolve().parent.parent.parent / "dashboard_data"


@st.cache_data
def load():
    inv = pd.read_csv(DATA / "inventory_recommendations.csv")
    comp = pd.read_csv(DATA / "inventory_policy_comparison.csv", index_col=0)
    return inv, comp


inv, comp = load()

st.title("📦 Inventory Optimization")
st.warning("Stock levels and supplier lead times are **simulated**, because the dataset has no inventory "
           "records. Demand, the forecast growth factor and the formulas are real.")

inv["OrderValue"] = inv["OrderQty"] * inv["UnitPrice"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Products analysed (top 30 by revenue)", len(inv))
k2.metric("Products to reorder", int((inv["Status"] == "Reorder").sum()))
k3.metric("Suggested order units", f"{inv['OrderQty'].sum():,.0f}")
k4.metric("Suggested order value", f"£{inv['OrderValue'].sum():,.0f}")

st.subheader("Reorder recommendations")
only_reorder = st.checkbox("Show only products that need reordering", value=True)
statuses = sorted(inv["Status"].unique())
chosen = st.multiselect("Status", statuses, default=statuses)
view = inv[inv["Status"].isin(chosen)]
if only_reorder:
    view = view[view["Status"] == "Reorder"]

cols = ["StockCode", "Description", "UnitPrice", "AvgDailyDemand", "LeadTime", "CurrentStock",
        "SafetyStock", "ReorderPoint", "EOQ", "Status", "OrderQty"]
st.dataframe(view[cols].round(1), hide_index=True)
st.download_button("Download recommendations (CSV)", view[cols].to_csv(index=False).encode(),
                   "retailpulse_reorder.csv", "text/csv")

st.subheader("Current stock as % of reorder point")
st.caption("Below 100% means stock is under the reorder point, so the product should be reordered.")
coverage = (view["CurrentStock"] / view["ReorderPoint"] * 100).round(0)
coverage.index = view["StockCode"].astype(str)
st.bar_chart(coverage.rename("Stock as % of reorder point"))

with st.expander("How the numbers are calculated"):
    st.markdown("""
- **Safety stock** = 1.65 x demand standard deviation x sqrt(lead time), a buffer for a 95% service level.
- **Reorder point** = average daily demand x lead time + safety stock.
- **EOQ** (economic order quantity) = sqrt(2 x annual demand x order cost / (holding rate x unit price)),
  with an assumed order cost of £50 and a 20% holding rate.
- **Status**: *Reorder* if stock is at or below the reorder point; *Overstock* if it is above the reorder point plus one EOQ.
- Daily demand uses the last 90 days before the test window, scaled by the forecast growth factor (1.176).
""")

st.subheader("Does the optimized policy help? (30-day simulation on real demand)")
st.caption("Compares a naive policy (no safety stock, 14 days of demand per order) with the optimized policy, "
           "using the same simulated starting stock and the actual demand from 9 Nov to 8 Dec 2011.")
st.dataframe(comp.round(1))
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Unmet demand (%)**")
    st.bar_chart(comp.loc[["Unmet demand %"], ["Naive", "Optimized"]].T)
with c2:
    st.markdown("**Total cost (£)**")
    st.bar_chart(comp.loc[["Total cost £"], ["Naive", "Optimized"]].T)
st.caption("Across three random seeds, the optimized policy cut unmet demand by 46% to 78% and total cost by "
           "41% to 69%, while average inventory roughly doubled (+120% to +128%). Assumed margin lost per unmet "
           "sale: 40%. These figures demonstrate the method; they are not savings for a real retailer.")