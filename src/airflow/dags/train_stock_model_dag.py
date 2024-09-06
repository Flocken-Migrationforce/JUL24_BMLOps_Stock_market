from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import sys

# Adding the directory containing app.py to the path to make it importable
sys.path.append('/Users/mehdienrahimi/JUL24_BMLOps_Stock_market/src')

# Importing the train_model function directly from app
from app import train_model

def train_wrapper():
    # Calling the train_model function directly; no need to import here since it's done globally
    train_model()

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'train_stock_model',
    default_args=default_args,
    description='DAG for training stock model',
    schedule_interval=timedelta(days=1),
)

train_task = PythonOperator(
    task_id='train_stock_model_task',
    python_callable=train_wrapper,
    dag=dag,
)