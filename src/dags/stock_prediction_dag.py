# dags/stock_prediction_dag.py

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from model import preprocess_data, prepare_datasets, create_model, train_model, validate_model, predict_prices
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 8, 23),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_prediction',
    default_args=default_args,
    description='A DAG for stock price prediction',
    schedule_interval=timedelta(days=1),
)

def preprocess_and_prepare_data(**kwargs):
    symbol = kwargs['dag_run'].conf.get('symbol', 'GOOGL')
    scaled_data, scaler, stock_prices_df = preprocess_data(symbol)
    x_train, y_train, x_val, y_val = prepare_datasets(scaled_data)
    kwargs['ti'].xcom_push(key='preprocessed_data', value=(scaled_data, scaler, x_train, y_train, x_val, y_val))

def train_and_validate_model(**kwargs):
    symbol = kwargs['dag_run'].conf.get('symbol', 'GOOGL')
    scaled_data, scaler, x_train, y_train, x_val, y_val = kwargs['ti'].xcom_pull(key='preprocessed_data', task_ids='preprocess_task')
    model = create_model()
    train_model(model, x_train, y_train)
    rmse, mae, mape, predictions_val, y_val = validate_model(model, x_val, y_val, scaler)
    model_path = f'models/{symbol}_prediction.h5'
    model.save(model_path)
    kwargs['ti'].xcom_push(key='metrics', value=(rmse, mae, mape))

def make_predictions(**kwargs):
    symbol = kwargs['dag_run'].conf.get('symbol', 'GOOGL')
    prediction_days = kwargs['dag_run'].conf.get('prediction_days', 7)
    scaled_data, scaler, _, _, _, _ = kwargs['ti'].xcom_pull(key='preprocessed_data', task_ids='preprocess_task')
    model_path = f'models/{symbol}_prediction.h5'
    model = load_model(model_path)
    predicted_prices = predict_prices(model, scaled_data, scaler, prediction_days)
    kwargs['ti'].xcom_push(key='predictions', value=predicted_prices.tolist())

preprocess_task = PythonOperator(
    task_id='preprocess_task',
    python_callable=preprocess_and_prepare_data,
    provide_context=True,
    dag=dag,
)

train_validate_task = PythonOperator(
    task_id='train_validate_task',
    python_callable=train_and_validate_model,
    provide_context=True,
    dag=dag,
)

predict_task = PythonOperator(
    task_id='predict_task',
    python_callable=make_predictions,
    provide_context=True,
    dag=dag,
)

preprocess_task >> train_validate_task >> predict_task