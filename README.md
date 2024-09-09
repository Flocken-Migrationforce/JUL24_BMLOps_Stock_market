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







### Disclaimer:
Work was done and completed between 24.07.2024 and 10.09.2024. <br>
This project is the practical part of the [AI / Machine Learning Engineer training](https://datascientest.com/en/machine-learning-engineer-course) from [DataScientest](https://datascientest.com/), cohort of July 2024 Data Scientist Bootcamp.
<br>All rights reserved by the authors. © 2024  <br>
