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
1. [📈 Introduction and Target](#-introduction-and-target)
2. [Project Organization](#project-organization)
3. [🎬 Getting Started for Developers](#-getting-started-for-developers)
   - [Set up the app](#set-up-the-app)
   - [Use the app](#use-the-app)
4. [Airflow Scheduling Usage](#airflow-scheduling-usage)
5. [MLflow](#mlflow)
6. [Prometheus](#prometheus)
7. [Grafana](#grafana)
8. [Disclaimer](#disclaimer)

## Architecture

![Architecture](https://raw.githubusercontent.com/Flocken-Migrationforce/JUL24_BMLOps_Stock_market/master/reports/figures/ML%20platform%20architecture_schema.png)
## Project Organization

    ├── docker-compose.yml    <- Docker Compose configuration.
    ├── LICENSE
    ├── README.md             <- The top-level README for developers using this project.
    ├── requirements.txt      <- Dependency requirements file.
    │
    ├── airflow               <- Airflow for scheduling model training.   
    │
    ├── app                   <- FastAPI application files.
    │
    ├── logs                  <- Logs from training and predicting
    │
    ├── models                <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks             <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                            the creator's initials, and a short `-` delimited description, e.g.
    │                            `1.0-jqp-initial-data-exploration`.
    │
    ├── references            <- Logos
    │
    ├── reports               <- Generated analyses, figures, etc.
    │
    ├── src                   <- Source code for use in this project.
    │   │
    │   ├── app.py            <- Main file to be executed.
    │   │
    │   ├── auth.py           <- File containing the password manager.
    │   │
    │   ├── data              <- Scripts to download or generate data, preprocessed data.
    │   │
    │   ├── models            <- Scripts to train models and use trained models to make predictions.
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── monitoring        <- Prometheus and alertmanager files.
    │   │
    │   ├── visualization     <- Scripts to create exploratory and results-oriented visualizations.
    │   │   └── visualize.py
    │   │
    │   ├── config            <- Describe the parameters used in train_model.py and predict_model.py.
    │   │
    │   ├── tests             <- Pytest files for unit and integration testing.
    │   │
    │   └── config            <- Describe the parameters used in train_model.py and predict_model.py


--------


## 🎬 Getting Started for Developers

### 🗃️ **Set up the app** 
#### 1. Clone the app's GitHub repository

```shell
git clone https://github.com/Flocken-Migrationforce/JUL24_BMLOps_Stock_market.git
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

## Airflow Scheduling
#### macOS :
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
#### Windows :
For Windows, you could use Docker Desktop with this [docker-compose.yml](https://airflow.apache.org/docs/apache-airflow/2.5.1/docker-compose.yaml) file and a file named ".env" with content
```Text
AIRFLOW_IMAGE_NAME=apache/airflow:2.4.2
AIRFLWO_UID=50000
```
Then, store these 2 files in the airflow directory of the project. Start Airflow by starting a Docker container containing the Airflow instance with terminal command
```Shell
docker-compose up -d
```

### Test : 
then you can go to UI in poer **:8081** and search for ID **train_aapl_model_daily**. You can start the scheduled training everyday at 9:00 AM. Note that airflow should be run in background all the time. 
or you can use the command below in src/airflow/dags/ PATH to test: 

```Shell
    python -c "from stock_prediction_dag import train_aapl; train_aapl()"
```

## MLflow
MLflow has to be started on port 8082, using the following command:

```shell
   mlflow server --host 0.0.0.0 --port 8082
```

will start automatically with app.py. you can go to "http://localhost:8082/" and track the experiment called **Stock Prediction LSTM**. 
few queries to test in shell. 

 ```shell
   curl -X POST "http://localhost:8000/train/AAPL" -H "Content-Type: application/json" -d '{"symbol": "AAPL"}' # track Endpoint: /train/{stocksymbol} , AAPL in this case

    curl -X GET "http://localhost:8000/users/" -H "Authorization: Bearer <your_access_token>" # get the list of users

```
Now, you can find the tracked information to the training run.

## Prometheus
1. Make sure you have Prometheus installed. Copy the execution files into src/monitoring.
2. in case it did not run automatically use run (this based in iOS, macOS):
   cd src/monitoring
   
   macOS:
   ```shell
   prometheus --config.file=prometheus.yml
   ```
   Windows:
   ```shell
   ./prometheus.exe --config.file=prometheus.yml 
   ```
4. Here are few queries to be tested in Prometheus.
   ```Shell
   up{job="fastapi"} # List All Available Metrics # This will return a basic status check to ensure Prometheus is scraping from your FastAPI instance.
   http_requests_total # count of total HTTP requests handled by your FastAPI app
   http_requests_total{handler="/train", method="POST"} # 
   ```


## Grafana 
1. Install Grafana on your machine.
2. You can use a Docker container to start Grafana, or start the grafana.exe file in your installation directory.
   grafana.
   ```shell
   bin\grafana-server.exe
   ```
   Or in a Docker container:
   ```shell
   docker run -d -p 3000:3000 grafana/grafana
   ```

4. Open in browser localhost:3000 to interact with the Grafana UI.
5. In the left sidebar, click on Configuration (gear icon) and then Data Sources.
      - Select Prometheus from the list of data sources.
      - In the URL field, enter: http://localhost:9090
         - Or for Prometheus run in a Docker container, enter: http://host.docker.internal:9090
6. Create a Dashboard in Grafana
    - You will see a graph or time series based on the query.
    - Adjust the visualization type (graph, gauge, table, etc.) using the options in the right sidebar.
    - Click Apply to add this panel to your dashboard.
    - you can use few queries from here to test:
 
| **Metric**                              | **Prometheus Query**                                                                                             | **Explanation**                                                                                                      |
|------------------------------------------|-------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| **Total Requests per Endpoint**          | `sum by (handler) (http_requests_total)`                                                                          | Shows the total number of HTTP requests to each endpoint. Useful to track traffic across different endpoints.         |
| **Total POST Requests to Training**      | `sum(http_requests_total{handler="/train/{stocksymbol}", method="POST"})`                                          | Counts the total number of POST requests to the `/train/{stocksymbol}` endpoint. Useful for tracking training activity.|
| **Total GET Requests to Metrics**        | `sum(http_requests_total{handler="/metrics", method="GET"})`                                                       | Counts the total number of GET requests to the `/metrics` endpoint. Useful to track how often Prometheus scrapes metrics.|
| **Total User Registration Requests**     | `sum(http_requests_total{handler="/users/register", method="POST"})`                                               | Counts the total number of user registration requests to the `/users/register` endpoint.                             |
| **Request Duration per Endpoint**        | `sum by (handler) (http_request_duration_seconds_sum) / sum by (handler) (http_request_duration_seconds_count)`     | Calculates the average duration for handling requests per endpoint. This is useful for identifying slow endpoints.    |
| **Total Number of Garbage Collections**  | `sum(python_gc_collections_total)`                                                                                 | Tracks the total number of times garbage collection was triggered in the application. Useful for memory management.   |
| **Garbage Collections per Generation**   | `sum by (generation) (python_gc_collections_total)`                                                                | Shows garbage collection events by generation. Helps identify memory usage patterns and potential memory leaks.       |
| **Objects Collected During GC**          | `sum by (generation) (python_gc_objects_collected_total)`                                                          | Shows how many objects were collected during garbage collection, broken down by generation.                           |
| **Uncollectable Objects in GC**          | `sum(python_gc_objects_uncollectable_total)`                                                                       | Tracks uncollectable objects during garbage collection. High values could indicate memory leaks.                      |
| **Total Response Size per Endpoint**     | `sum by (handler) (http_response_size_bytes_sum)`                                                                  | Monitors the total size of responses sent by each endpoint. Useful to understand bandwidth usage and performance.     |




### Disclaimer:
Work was done and completed between 24.07.2024 and 10.09.2024. <br>
This project is the practical part of the [AI / Machine Learning Engineer training](https://datascientest.com/en/machine-learning-engineer-course) from [DataScientest](https://datascientest.com/), cohort of July 2024 Data Scientist Bootcamp.
<br>All rights reserved by the authors. © 2024  <br>
