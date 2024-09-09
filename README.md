# 📈💲Stock Market Prediction

## 🧺 Introduction and Target

In this MLOps project, we have developed an AI-ML model (LSTM model) to predict the evolution of shares of Apple (AAPL) and Google (GOOGL) at the stock market.
An API was created and deployed, allowing access and interaction to our stock prediction model for the interested audience with an affinity for finance. Retraining of the model is performed daily at 9 a.m., to ensure accurate predictions.

We are Machine Learning Engineers :

<a href="https://github.com/Flocken-Migrationforce"><img src="https://github.com/Flocken-Migrationforce.png" width="50px" alt="Fabian Flocken" style="border-radius:50%"></a>
<a href="https://github.com/mirmehdi"><img src="https://github.com/mirmehdi.png" width="50px" alt="Mir Mehdi Seyedebrahimi" style="border-radius:50%"></a>
<a href="https://github.com/LeoLoeff"><img src="https://github.com/LeoLoeff.png" width="50px" alt="Leonhard Löffler" style="border-radius:50%"></a>

[Fabian Flocken](https://www.linkedin.com/in/fabian-flocken-0638a9315)<br>
[Mir Mehdi Seyedebrahimi](https://www.linkedin.com/in/mirmehdiseyedebrahimi/)<br>
[Leonhard Löffler](https://www.linkedin.com/in/leonhard-loeffler/)<br>

In this project, we developed a MLOps solution to deploy, monitor and update a self-built LSTM model, using Neural Networks to predict stock market prices of famous Tech companies.

## Table of Contents
1. [Project Overview](reports/project_overview.md)
2. [Data Collection](reports/data_collection.md)
3. [Data Preprocessing](reports/data_preprocessing.md)
4. [Exploratory Data Analysis](notebooks/eda.ipynb)
5. [Model Selection](reports/model_selection.md)
6. [Model Training](reports/model_training.md)
7. [Model Evaluation](reports/model_evaluation.md)
8. [Deployment](reports/deployment.md)
9. [Monitoring and Maintenance](reports/monitoring_and_maintenance.md)



--unspecific Template content below--
<LEO>
Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── logs               <- Logs from training and predicting
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


# Stock Market Prediction Project

This project aims to develop an AI/ML model to predict stock market prices and trends.

## Table of Contents
1. [Project Overview](reports/project_overview.md)
2. [Data Collection](reports/data_collection.md)
3. [Data Preprocessing](reports/data_preprocessing.md)
4. [Exploratory Data Analysis](notebooks/eda.ipynb)
5. [Model Selection](reports/model_selection.md)
6. [Model Training](reports/model_training.md)
7. [Model Evaluation](reports/model_evaluation.md)
8. [Deployment](reports/deployment.md)
9. [Monitoring and Maintenance](reports/monitoring_and_maintenance.md)


## 🎬 Getting Started for Developers

### 🗃️ **Set up the app** 
#### 1. Clone the app's GitHub repository

```shell
[git clone https://github.com/omarchoa/dec23_mlops_accidents.git](https://github.com/Flocken-Migrationforce/JUL24_BMLOps_Stock_market.git)
```

#### 2. Set up a Python virtual environment

Using Python's `venv` module, with `sword` as the virtual environment name, run the following commands:

```shell
python -m venv sword
chmod +x ./sword/bin/activate
source ./sword/bin/activate
```
#### 3. Install the app's global dependencies

From the directory into which you cloned the GitHub repository, go to src direction and  run the following command:

```shell
cd .../Path to src/
pip install -r requirements.txt
```


## ⚙️ **Use the app**
In ./src folder and run the app.py (fast api). The applicatoin will automatoically start the app in port 8000. 

```shell
cd .../Path to src/
python app.py

curl http://localhost:8000/docs
```
Visit http://localhost:8000/docs to easily create a new user, set a password, and choose between premium or basic subscription plans. You can also update user information at any time. Please ensure you log in with the correct credentials and provide a valid stock symbol (e.g., AAPL) when using the training feature. Note that only premium subscribers have access to stock market predictions."

## Airflow Schaduling Usage
Note that the airflow creation here is defined in mac os. It could be different for other os systems. 

1. install apache-airflow
```shell
   pip install apache-airflow
```
2. initiate database
 ```shell
   airflow db init
```  
3. before starting go to **airflow.cfg** (you can search for this file using **airflow info** in the shell) and be sure that the path inside **airflow.cfg** is addressing the python file "PATH TO /src/airflow/dagsstock_prediction_dag.py".
4. then run Apache airflow in port 8081. Note that this terminal will be occupied. 
```shell
   airflow webserver -p 8081 
```
5. run scheduler in another terminal. Note that this terminal will be occupied. 

```shell
   airflow scheduler 
```

### Test: 
then you can go to UI in poer **:8081** and search for ID **train_aapl_model_daily**. You can start the scheduled training everyday at 9:00 AM. Note that airflow should be run in background all the time. 
or you can use the command below in src/airflow/dags/ PATH to test: 

```Shell
    python -c "from stock_prediction_dag import train_aapl; train_aapl()"
```

## MLflow
MLflow will start automatically with app.py. you can go to "http://localhost:8082/" and track the experiment called **Stock Prediction LSTM**. 
few queries to test in shell. 

 ```shell
   curl -X POST "http://localhost:8000/train/AAPL" -H "Content-Type: application/json" -d '{"symbol": "AAPL"}' # track Endpoint: /train/{stocksymbol} , AAPL in this case

    curl -X GET "http://localhost:8000/users/" -H "Authorization: Bearer <your_access_token>" # get the list of users

```


## Prometheus
1. Simply open port **:9090** and use prometheus UI
2. in case it did not run automatically use run (this based in ios, mac):
   prometheus --config.file=/path/to/prometheus.yml
4. Here are few queries to be tested in promotheus.
   ```Shell
   fastapi_requests_total{instance="localhost:8000"} 
    # Total number of HTTP requests received by your FastAPI app running on localhost:8000.
    
    fastapi_requests_total{handler="/train/{stocksymbol}", method="POST", instance="localhost:8000"} 
    # Total number of POST requests made to the '/train/{stocksymbol}' endpoint in your FastAPI app on localhost:8000.
    
    fastapi_requests_total{handler="/predict/{stocksymbol}", method="POST", instance="localhost:8000"} 
    # Total number of POST requests made to the '/predict/{stocksymbol}' endpoint in your FastAPI app on localhost:8000.
    
    fastapi_request_duration_seconds_sum{instance="localhost:8000"} 
    # The total sum of time spent handling requests by your FastAPI app on localhost:8000.
    
    fastapi_request_duration_seconds_count{instance="localhost:8000"} 
    # The count of requests handled by FastAPI, useful when calculating average request duration.
    
    rate(fastapi_requests_total{handler="/train/{stocksymbol}"}[5m]) 
    # Rate of requests to the '/train/{stocksymbol}' endpoint over the last 5 minutes, useful for tracking traffic to a specific endpoint.
    
    fastapi_request_duration_seconds_sum{handler="/train/{stocksymbol}", instance="localhost:8000"} / 
    fastapi_request_duration_seconds_count{handler="/train/{stocksymbol}", instance="localhost:8000"} 
    # Average duration (latency) of POST requests to the '/train/{stocksymbol}' endpoint on localhost:8000.


   ```


## Grafana 
- you can use UI in port **:3000**
- In the left sidebar, click on Configuration (gear icon) and then Data Sources.
      - Select Prometheus from the list of data sources.
      - In the URL field, enter: http://localhost:9090
- Create a Dashboard in Grafana
    - You will see a graph or time series based on the query.
    - Adjust the visualization type (graph, gauge, table, etc.) using the options in the right sidebar.
    - Click Apply to add this panel to your dashboard.
    - you can use few queries from here to test:
 
```
fastapi_requests_total{instance="localhost:8000"}
# Total number of HTTP requests made to your FastAPI application running on localhost:8000.

fastapi_requests_total{handler="/train/{stocksymbol}", method="POST", instance="localhost:8000"}
# Total number of POST requests made to the '/train/{stocksymbol}' endpoint in your FastAPI app running on localhost:8000.

rate(fastapi_requests_total[5m])
# The rate of HTTP requests to all FastAPI endpoints over the last 5 minutes.

rate(fastapi_requests_total{handler="/predict/{stocksymbol}"}[5m])
# The rate of POST requests made to the '/predict/{stocksymbol}' endpoint over the last 5 minutes.

rate(fastapi_request_duration_seconds_sum[5m]) / rate(fastapi_request_duration_seconds_count[5m])
# Average duration (latency) of HTTP requests to your FastAPI application over the last 5 minutes.

fastapi_request_duration_seconds_sum{handler="/train/{stocksymbol}", instance="localhost:8000"} / 
fastapi_request_duration_seconds_count{handler="/train/{stocksymbol}", instance="localhost:8000"}
# Average duration (latency) of POST requests to the '/train/{stocksymbol}' endpoint.

process_resident_memory_bytes{job="fastapi", instance="localhost:8000"}
# Amount of resident memory used by the FastAPI application running on localhost:8000.

node_memory_MemAvailable_bytes{instance="localhost:9100"}
# Total available memory in bytes on the system where Node Exporter is running (e.g., localhost:9100).

node_filesystem_free_bytes{mountpoint="/", instance="localhost:9100"}
# Free disk space in bytes available on the root (/) mountpoint of the system monitored by Node Exporter.

rate(node_network_receive_bytes_total[5m])
# Rate of network traffic (bytes received) over the last 5 minutes on all network interfaces.

up{instance="localhost:9090"}
# Whether the Prometheus instance running on localhost:9090

```



### Disclaimer:
Work was done and completed between 24.07.2024 and 10.09.2024. <br>
This project is the practical part of the [AI / Machine Learning Engineer training](https://datascientest.com/en/machine-learning-engineer-course) from [DataScientest](https://datascientest.com/), cohort of July 2024 Data Scientist Bootcamp.
<br>All rights reserved by the authors. © 2024  <br>
