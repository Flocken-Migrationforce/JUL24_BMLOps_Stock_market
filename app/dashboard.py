# Dashboard at port 8080
# Fabian
# Version 0.1
# 2408011227

# app/dashboard.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import requests
import docker

# Initialize the Dash app
dash_app = dash.Dash(__name__)

# Assume these are the endpoints of your FastAPI application
FASTAPI_URL = "http://localhost:8000"
AIRFLOW_URL = "http://localhost:8080/api/v1"

# Initialize Docker client
docker_client = docker.from_env()

# Layout of the dashboard
dash_app.layout = html.Div([
	html.H1("FastAPI and Airflow Dashboard"),

	html.Div([
		html.H2("Airflow DAGs"),
		dcc.Graph(id='dag-status-graph'),
		dcc.Interval(
			id='interval-component',
			interval=30 * 1000,  # in milliseconds
			n_intervals=0
		)
	]),

	html.Div([
		html.H2("Docker Containers"),
		html.Ul(id='container-list')
	]),

	html.Div([
		html.H2("Task Queue"),
		html.Ul(id='task-queue')
	])
])


@dash_app.callback(
	Output('dag-status-graph', 'figure'),
	Input('interval-component', 'n_intervals')
)
def update_dag_status(n):
	# In a real scenario, you would fetch this data from your Airflow API
	# For this example, we'll use dummy data
	dag_status = pd.DataFrame({
		'DAG': ['dag1', 'dag2', 'dag3'],
		'Status': ['success', 'running', 'failed']
	})
	fig = px.bar(dag_status, x='DAG', y='Status', color='Status',
	             color_discrete_map={'success': 'green', 'running': 'blue', 'failed': 'red'})
	return fig


@dash_app.callback(
	Output('container-list', 'children'),
	Input('interval-component', 'n_intervals')
)
def update_container_list(n):
	containers = docker_client.containers.list()
	return [html.Li(f"{container.name}: {container.status}") for container in containers]


@dash_app.callback(
	Output('task-queue', 'children'),
	Input('interval-component', 'n_intervals')
)
def update_task_queue(n):
	# In a real scenario, you would fetch this data from your Redis queue
	# For this example, we'll use dummy data
	tasks = ['Task 1', 'Task 2', 'Task 3']
	return [html.Li(task) for task in tasks]


if __name__ == '__main__':
	dash_app.run_server(debug=True)