"""Retraining check for the churn model: load -> validate -> drift check -> retrain -> log."""
import sys
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

DATA = "data/processed/customers_clean.csv"
REF_CUTOFF = pd.Timestamp("2011-09-09")
DRIFT_THRESHOLD = 0.30
PARAMS = {  # tuned with Optuna on Day 11
    "n_estimators": 411, "max_depth": 3, "learning_rate": 0.0138,
    "subsample": 0.61, "colsample_bytree": 0.89, "min_child_weight": 8,
}


def build_features(df, cutoff):
    past = df[df["InvoiceDate"] < cutoff]
    f = past.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (cutoff - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        Products=("StockCode", "nunique"),
        First=("InvoiceDate", "min"),
    )
    f["Tenure"] = (cutoff - f["First"]).dt.days
    f["AvgOrderValue"] = f["Monetary"] / f["Frequency"]
    f = f.drop(columns="First")
    f = f[f["Recency"] <= 180]

    inv = (past.drop_duplicates(["Customer ID", "Invoice"])
               [["Customer ID", "Invoice", "InvoiceDate"]]
               .sort_values(["Customer ID", "InvoiceDate"]))
    gap = inv.groupby("Customer ID")["InvoiceDate"].apply(lambda s: s.diff().dt.days.mean())
    f["AvgGapDays"] = gap.reindex(f.index).fillna(gap.median())
    f["OverdueRatio"] = f["Recency"] / (f["AvgGapDays"] + 1)
    for d in (30, 90):
        r = inv[inv["InvoiceDate"] >= cutoff - pd.Timedelta(days=d)]
        f[f"Orders_last{d}d"] = r.groupby("Customer ID")["Invoice"].nunique().reindex(f.index).fillna(0)
    f["RecentShare90"] = f["Orders_last90d"] / f["Frequency"]
    return f


def add_label(f, df, cutoff, days=90):
    future = df[(df["InvoiceDate"] >= cutoff) &
                (df["InvoiceDate"] < cutoff + pd.Timedelta(days=days))]
    f = f.copy()
    f["Churn"] = (~f.index.isin(set(future["Customer ID"]))).astype(int)
    return f


def validate(df):
    assert df["Customer ID"].notna().all(), "null Customer ID"
    assert (df["Price"] > 0).all(), "non-positive price"
    assert (df["Quantity"] > 0).all(), "non-positive quantity"
    assert (df["Revenue"] > 0).all(), "non-positive revenue"
    print("Validation passed")


def drift_share(ref, cur):
    cols = ref.columns.tolist()
    dd = DataDefinition(numerical_columns=cols)
    ds = lambda d: Dataset.from_pandas(d[cols].reset_index(drop=True), data_definition=dd)
    result = Report([DataDriftPreset()]).run(current_data=ds(cur), reference_data=ds(ref))
    result.save_html("reports/drift/drift_latest.html")
    return result.dict()["metrics"][0]["value"]["share"]


def retrain(df):
    cutoff = (df["InvoiceDate"].max() - pd.Timedelta(days=90)).normalize()
    data = add_label(build_features(df, cutoff), df, cutoff)
    X, y = data.drop(columns="Churn"), data["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBClassifier(**PARAMS, eval_metric="auc", random_state=42).fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("retailpulse-churn")
    with mlflow.start_run(run_name="scheduled_retrain"):
        mlflow.log_params(PARAMS)
        mlflow.log_param("cutoff", str(cutoff.date()))
        mlflow.log_metric("test_auc", auc)
    print(f"Retrained. cutoff={cutoff.date()} test AUC={auc:.3f}")


if __name__ == "__main__":
    df = pd.read_csv(DATA, parse_dates=["InvoiceDate"])
    validate(df)

    ref = build_features(df, REF_CUTOFF)
    cur = build_features(df, df["InvoiceDate"].max() + pd.Timedelta(days=1))
    if "--simulate-drift" in sys.argv:
        cur = cur.copy()
        cur["Recency"] = cur["Recency"] * 1.5 + 20
        cur["Monetary"] = cur["Monetary"] * 0.6
        cur["Frequency"] = (cur["Frequency"] * 0.7).clip(lower=1).round()

    share = drift_share(ref, cur)
    print(f"Drift share: {share:.0%} (threshold {DRIFT_THRESHOLD:.0%})")
    if share > DRIFT_THRESHOLD:
        print("Drift alert -> retraining")
        retrain(df)
    else:
        print("No significant drift -> no retraining")