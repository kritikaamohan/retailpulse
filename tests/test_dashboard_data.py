from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parent.parent / "dashboard_data"


def read(name, **kwargs):
    path = DATA / name
    if not path.exists():
        pytest.fail(f"{name} is missing from dashboard_data")
    return pd.read_csv(path, **kwargs)


def test_daily_country_is_clean():
    df = read("daily_country.csv")
    assert not df.isna().any().any()
    assert (df["Revenue"] > 0).all()
    assert (df["Units"] > 0).all()


def test_forecast_has_30_days_and_no_negatives():
    fc = read("final_forecast.csv", index_col=0, parse_dates=True).iloc[:, 0]
    assert len(fc) == 30
    assert (fc >= 0).all()


def test_forecast_is_zero_on_saturdays():
    fc = read("final_forecast.csv", index_col=0, parse_dates=True).iloc[:, 0]
    assert (fc[fc.index.dayofweek == 5] == 0).all()


def test_segments_are_complete():
    seg = read("rfm_segments.csv")
    assert seg["Segment"].nunique() == 6
    assert seg["Customer ID"].is_unique
    assert not seg[["Recency", "Frequency", "Monetary"]].isna().any().any()


def test_churn_probabilities_are_valid():
    churn = read("churn_scores.csv")
    assert churn["Customer ID"].is_unique
    assert churn["ChurnProb"].between(0, 1).all()


def test_inventory_logic_is_consistent():
    inv = read("inventory_recommendations.csv")
    assert (inv["ReorderPoint"] >= inv["SafetyStock"]).all()
    assert (inv["OrderQty"] >= 0).all()
    assert set(inv["Status"]) <= {"Reorder", "Healthy", "Overstock"}
    # only products marked Reorder get an order quantity
    assert (inv.loc[inv["Status"] != "Reorder", "OrderQty"] == 0).all()


def test_drift_summary_shares_are_valid():
    drift = read("drift_summary.csv")
    assert drift["share"].between(0, 1).all()
    assert set(drift["alert"].astype(bool)) <= {True, False}