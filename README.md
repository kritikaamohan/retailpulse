# RetailPulse: AI-Powered Customer Analytics & Demand Forecasting Platform

![CI](https://github.com/kritikaamohan/retailpulse/actions/workflows/ci.yml/badge.svg)

**TODO (2-3 sentences in your own words): what problem does RetailPulse solve, and who would use it?**

- **Live demo:** https://retailpulse-kritikamohan.streamlit.app/
- **Demo video:** TODO add the link after you upload it
- **Project report:** TODO add the PDF to the repo (or a link)

![Overview](reports/figures/20_live_overview.jpeg)

## Features

| Page | What it does |
|---|---|
| Overview | KPIs, daily revenue, top countries and products, with date and country filters |
| Demand Forecast | 30-day revenue forecast, model comparison, promo-lift what-if slider |
| Segments & Churn | 6 RFM customer segments; churn risk list with SHAP explanations |
| Inventory | Reorder points, safety stock, EOQ and a policy simulation |
| Monitoring | Evidently drift reports and a drift simulator |

**TODO (1 sentence): what are you proudest of in this project?**

## Architecture

![Architecture](reports/figures/00_architecture.png)

The dashboard only reads small precomputed files from `dashboard_data/`, so it opens fast and needs no secrets or paid services.

## Data

Online Retail II (UCI): a UK online retailer, December 2009 to December 2011, 1,067,371 transaction rows.

| Cleaning step | Rows removed |
|---|---|
| Exact duplicates | 34,335 |
| Cancelled invoices | 19,104 |
| Price <= 0 (includes stock adjustments) | 6,019 |
| Non-product stock codes | 4,699 |
| **Rows kept** | **1,003,214** of 1,067,371 |

About 23% of rows have no customer ID. They are kept for demand forecasting and excluded from customer models (5,852 customers).

**TODO (2-3 sentences): why you made these cleaning choices.**

Stock levels and supplier lead times on the Inventory page are **simulated**, because the dataset has no inventory records. The dataset also has no cost data, so only revenue is analyzed.

## Results

**Demand forecasting** (30-day test window, 9 Nov to 8 Dec 2011):

| Model | MAPE (trading days) | WAPE |
|---|---|---|
| Seasonal naive | 29.37% | 25.53% |
| Prophet baseline | 20.83% | 26.06% |
| LSTM | 33.32% | 36.31% |
| Ensemble (50/50) | 21.40% | 27.48% |
| Prophet + Saturday rule | 20.83% | 19.80% |
| **Prophet tuned + Saturday rule (final)** | **19.34%** | **19.49%** |

MAPE is computed on trading days only, because the shop is closed on Saturdays and the actual value is zero. On three earlier 30-day validation windows the daily MAPE averages about 32%, so the test result is on the favourable side. Weekly (7-day total) MAPE is 9.9% on the test window and 17.5% on average across the validation windows (4 weeks each, so a rough indication).

**Churn prediction** (no purchase in the 90 days after the cutoff date):

| Evaluation | AUC | Precision@top 20% |
|---|---|---|
| Random split, test set (cutoff 2011-09-09) | 0.714 | 0.658 |
| Cross-validation (training data) | 0.780 | n/a |
| Out-of-time: trained on 12 earlier cutoffs, tested on 2011-09-09 | 0.768 | 0.686 |
| Project targets | 0.88 | 0.75 |

The base churn rate is 38.5%, so precision@top 20% of 0.686 is about 1.8 times better than random.

**Customer segments:** 5,852 customers in 6 segments. The 472 Champions (8.1%) account for 54.4% of the revenue from identified customers.

**Inventory (simulated stock):** in a 30-day simulation on real demand, the optimized reorder policy cut unmet demand by 46% to 78% across three random seeds, while average inventory roughly doubled.

**Drift monitoring:** 29% of features flagged on real data (no retraining); 71% on simulated drift (retraining triggered).

**TODO (2-3 sentences, honest): which targets were missed (forecast MAPE, churn AUC and precision) and why.**

## Tech stack

Python 3.11, pandas, scikit-learn, Prophet, PyTorch Lightning, XGBoost, SHAP, Optuna, MLflow, Evidently, Great Expectations, Streamlit, Docker, GitHub Actions, pytest, ruff.

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
requirements.txt     minimal packages for the Streamlit app (used for deployment)
requirements-dev.txt full environment used for training and notebooks
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux
pip install -r requirements-dev.txt
streamlit run app/main.py
```

To rebuild the results, download [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) as `data/raw/online_retail_II.csv`, run the notebooks in order (01 to 06), then:

```bash
python src/make_dashboard_data.py
pytest -q
python src/retrain.py
```

## Limitations

- Stock levels and lead times are simulated, because the data has no inventory records.
- Forecast MAPE (19.34% on test, about 32% on earlier windows) and churn AUC (0.714 random split, 0.768 out-of-time) are below the project targets.
- Churn was evaluated at a single future cutoff date, in the busy autumn period.
- The what-if slider scales the forecast and is a scenario, not a trained promotion model.
- The Airflow DAG and Kubernetes manifests are written but were not run. Prometheus and Grafana were not built.

## Author

Kritika Mohan, October 2026
