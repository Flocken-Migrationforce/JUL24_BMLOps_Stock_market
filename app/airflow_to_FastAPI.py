# Airflow's REST API: Logic of Airflow integration into FastAPI
# Version 0.1
# Fabian
# 2408011209

import httpx
from fastapi import FastAPI

airflow_app = FastAPI()

AIRFLOW_API_URL = "http://localhost:8080/api/v1"
AIRFLOW_USERNAME = "your_airflow_username"
AIRFLOW_PASSWORD = "your_airflow_password"


@airflow_app.post("/trigger-dag/{dag_id}")
async def trigger_dag(dag_id: str):
	url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns"
	auth = httpx.BasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)

	async with httpx.AsyncClient() as client:
		response = await client.post(url, json={}, auth=auth)

	if response.status_code == 200:
		return {"message": f"DAG {dag_id} triggered successfully"}
	else:
		return {"error": f"Failed to trigger DAG. Status code: {response.status_code}"}