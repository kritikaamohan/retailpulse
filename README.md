# RetailPulse: AI-Powered Customer Analytics & Demand Forecasting Platform

![CI](https://github.com/kritikaamohan/retailpulse/actions/workflows/ci.yml/badge.svg)

<!-- WRITE: 2-3 sentences. What problem does this solve and for whom? -->

**Live demo:** https://retailpulse-kritikamohan.streamlit.app/

![Overview](reports/figures/20_live_overview.png)

## Features

| Page | What it does |
|---|---|
| Overview | KPIs, daily revenue, top countries and products, with date and country filters |
| Demand Forecast | 30-day revenue forecast, model comparison, promo-lift what-if slider |
| Segments & Churn | 6 RFM customer segments; churn risk list with SHAP explanations |
| Inventory | Reorder points, safety stock, EOQ and a policy simulation |
| Monitoring | Evidently drift reports and a drift simulator |

<!-- WRITE: one sentence on what you are proudest of -->

## Data

Online Retail II (UCI): a UK online retailer, December 2009 to December 2011.

| Cleaning step | Rows removed |
|---|---|
| Exact duplicates | 34,335 |
| Cancelled invoices | 19,104 |
| Price <= 0 (includes stock adjustments) | 6,019 |
| Non-product stock codes | 4,699 |
| **Rows kept** | **1,003,214** of 1,067,371 |

<!-- WRITE: why you made these cleaning choices, and that stock levels in the Inventory page are simulated -->

## Results

**Demand forecasting** (30-day test window, 9 Nov to 8 Dec 2011):

| Model | MAPE (trading days) | WAPE |
|---|---|---|
| Seasonal naive | 29.37% | 25.53% |
| Prophet baseline | 20.83% | 26.06% |
| LSTM | 33.32% | 36.31% |
| Ensemble (50/50) | 21.40% | 27.48% |
| **Prophet + Saturday rule** | **20.83%** | **19.80%** |

**Churn prediction** (no purchase in the 90 days after 2011-09-09):

| Metric | Result | PDF target |
|---|---|---|
| CV AUC | 0.780 | n/a |
| Test AUC | 0.714 | 0.88 |
| Precision@top 20% | 0.658 | 0.75 |

**Customer segments:** 5,852 customers in 6 segments. The 472 Champions (8.1%) account for 54.4% of the revenue from identified customers.

**Drift monitoring:** 29% of features flagged on real data (no retraining); 71% on simulated drift (retraining triggered).

<!-- WRITE: be honest. Say which targets were missed and why (zero-revenue Saturdays, small test set, limited signal in the data) -->

## Tech stack

Python 3.11, pandas, scikit-learn, Prophet, PyTorch Lightning, XGBoost, SHAP, Optuna, MLflow, Evidently, Great Expectations, Streamlit, Docker, GitHub Actions.

## Project structure

```
app/                 Streamlit app (main.py and pages/)
dashboard_data/      small result files the app reads
notebooks/           01_eda, 02_cleaning_features, 03_segmentation,
                     04_forecasting, 05_churn, 06_inventory
src/                 metrics.py, retrain.py, make_dashboard_data.py
dags/                Airflow DAG for scheduled retraining
tests/               pytest checks for metrics and saved results
k8s/                 Kubernetes manifests (written, not deployed)
reports/             figures and drift reports
Dockerfile           multi-stage image for the app
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app/main.py
```

To rebuild the results, download Online Retail II as `data/raw/online_retail_II.csv`, run the notebooks in order, then:

```bash
python src/make_dashboard_data.py
pytest -q
python src/retrain.py
```

## Limitations

- Stock levels and lead times are simulated, because the data has no inventory records.
- Forecast MAPE (20.83%) and churn AUC (0.714) are below the project targets.
- The Airflow DAG and Kubernetes manifests are written but were not run.

## Author

Kritika Mohan

<!-- WRITE: your name and the date -->
