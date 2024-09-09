# 📈💲Stock Market Prediction

## 🧺 Introduction and Target

In this MLOps project, we are currently developing AI/ML models to predict the evolution of shares of Apple (AAPL) and Google (GOOGL) at the stock market.
An API will be created and deployed, to allow access to and interaction with interested parties with an affinity for finance. Daily updates ensure the data to be up-to-date.

We are Machine Learning Engineers :

<a href="https://github.com/Flocken-Migrationforce"><img src="https://github.com/Flocken-Migrationforce.png" width="50px" alt="Fabian Flocken" style="border-radius:50%"></a>
<a href="https://github.com/mirmehdi"><img src="https://github.com/mirmehdi.png" width="50px" alt="Mir Mehdi Seyedebrahimi" style="border-radius:50%"></a>
<a href="https://github.com/LeoLoeff"><img src="https://github.com/LeoLoeff.png" width="50px" alt="Leonhard Löffler" style="border-radius:50%"></a>

[Fabian Flocken](https://www.linkedin.com/in/fabian-flocken-0638a9315)<br>
[Mir Mehdi Seyedebrahimi](https://www.linkedin.com/in/mirmehdi)<br>
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
   http_requests_total{handler="/metrics", instance="localhost:8000", job="fastapi", method="GET", status="2xx"} 
    # Total number of successful GET requests (status code 2xx) made to the '/metrics' endpoint 
    # from the FastAPI application running on localhost:8000.
    
    up{instance="localhost:9090", job="prometheus"} 
    # Whether the Prometheus server running on localhost:9090 is up (1 means up, 0 means down).
    
    process_cpu_seconds_total{job="fastapi", instance="localhost:8000"} 
    # Total CPU time used by the FastAPI application running on localhost:8000.
    
    node_memory_MemAvailable_bytes{instance="localhost:9100"} 
    # The amount of available memory on the machine monitored by Node Exporter running on localhost:9100.
    
    rate(http_requests_total[5m]) 
    # The rate of HTTP requests over the last 5 minutes for all endpoints.
    
    node_filesystem_free_bytes{mountpoint="/", instance="localhost:9100"} 
    # The amount of free disk space available on the root mount (/) of the machine monitored by Node Exporter on localhost:9100.
    
    scrape_duration_seconds{job="fastapi", instance="localhost:8000"} 
    # The duration (in seconds) that Prometheus took to scrape the FastAPI application's metrics on localhost:8000.
    
    node_network_receive_bytes_total{device="eth0", instance="localhost:9100"} 
    # Total number of bytes received over the 'eth0' network interface on the machine monitored by Node Exporter running on localhost:9100.
    
    go_gc_duration_seconds{job="prometheus"} 
    # The duration of garbage collection pauses in the Prometheus server.
    
    process_resident_memory_bytes{job="fastapi", instance="localhost:8000"} 
    # The amount of resident memory used by the FastAPI application running on localhost:8000.

   ```
    



### Disclaimer:
Work was done and completed between 24.07.2024 and 10.09.2024. <br>
This project is the practical part of the [AI / Machine Learning Engineer training](https://datascientest.com/en/machine-learning-engineer-course) from [DataScientest](https://datascientest.com/), cohort of July 2024 Data Scientist Bootcamp.
<br>All rights reserved by the authors. © 2024  <br>
