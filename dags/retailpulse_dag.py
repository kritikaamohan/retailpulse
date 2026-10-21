from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="retailpulse_retraining",
    start_date=datetime(2026, 10, 1),
    schedule="@weekly",
    catchup=False,
    tags=["retailpulse", "mlops"],
) as dag:
    BashOperator(
        task_id="validate_drift_retrain",
        bash_command="cd /opt/retailpulse && python src/retrain.py",
    )