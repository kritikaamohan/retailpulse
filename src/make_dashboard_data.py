"""Build small, pre-aggregated CSVs for the Streamlit app (run from the project root)."""
import os
import shutil
import pandas as pd

os.makedirs("dashboard_data", exist_ok=True)

sales = pd.read_csv("data/processed/sales_clean.csv", parse_dates=["InvoiceDate"])
sales["date"] = sales["InvoiceDate"].dt.normalize()

# 2011-12-09 is a partial day, so it is excluded (same rule as the forecasting series)
sales = sales[sales["date"] <= "2011-12-08"]

daily_country = (sales.groupby(["date", "Country"])
                 .agg(Revenue=("Revenue", "sum"),
                      Orders=("Invoice", "nunique"),
                      Units=("Quantity", "sum"))
                 .reset_index())
daily_country.to_csv("dashboard_data/daily_country.csv", index=False)

top = (sales.groupby("StockCode")
       .agg(Description=("Description", lambda s: s.mode().iloc[0]),
            Revenue=("Revenue", "sum"), Units=("Quantity", "sum"))
       .nlargest(15, "Revenue").reset_index())
top.to_csv("dashboard_data/top_products.csv", index=False)

static = pd.DataFrame({
    "customers": [sales["Customer ID"].nunique()],
    "products": [sales["StockCode"].nunique()],
})
static.to_csv("dashboard_data/static_kpis.csv", index=False)

for f in ["final_forecast.csv", "churn_scores.csv", "rfm_segments.csv",
          "inventory_recommendations.csv", "inventory_policy_comparison.csv",
          "drift_summary.csv", "daily_prepared.csv"]:
    src = os.path.join("data/processed", f)
    if os.path.exists(src):
        shutil.copy(src, "dashboard_data")
        print("copied", f)
    else:
        print("MISSING", f)

for f in os.listdir("dashboard_data"):
    print(f, round(os.path.getsize(os.path.join("dashboard_data", f)) / 1e6, 2), "MB")