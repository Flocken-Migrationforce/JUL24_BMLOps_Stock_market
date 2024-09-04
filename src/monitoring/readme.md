Created: MMS

ALARM: ensure they update the **alertmanager.yml** with your email credentials.


FastAPI Monitoring with Prometheus, Alertmanager, and Grafana
This project provides a setup for monitoring a FastAPI application using Prometheus, Alertmanager, and Grafana. Follow the instructions below to set up the monitoring stack and visualize the FastAPI metrics.

Prerequisites
Before starting, ensure the following tools are installed on your system:

Prometheus (>= 2.0)
Alertmanager (>= 0.22)
FastAPI (with prometheus_fastapi_instrumentator)
Grafana (optional but recommended for visualization)
Quick Setup Guide
1. Clone the Repository
First, clone this repository to your local machine:

bash
Copy code
git clone <repository-url>
cd <repository-folder>
2. FastAPI Application Setup
Ensure your Python environment is set up, and dependencies are installed. For example, you can use pip:

bash
Copy code
pip install -r requirements.txt
Ensure that the FastAPI app is configured to expose metrics at the /metrics endpoint. The provided FastAPI app already has this enabled with prometheus_fastapi_instrumentator.

To start the FastAPI application, run:

bash
Copy code
uvicorn app:app --reload
This should start your FastAPI app on http://localhost:8001. Ensure the /metrics endpoint is available by visiting http://localhost:8001/metrics.

3. Prometheus Setup
Prometheus is used to scrape metrics from FastAPI and monitor performance. The configuration file for Prometheus is located in the monitoring folder.

Start Prometheus
Prometheus will be set up to scrape metrics from both Prometheus and FastAPI. You can start Prometheus using the prometheus.yml configuration file provided:

bash
Copy code
/path/to/prometheus --config.file=./monitoring/prometheus.yml
Ensure that both prometheus.yml and alerts.yml are in the correct directory as configured.

Once Prometheus is running, open the Prometheus UI by visiting http://localhost:9090.

4. Alertmanager Setup
The Alertmanager handles sending alerts when certain conditions are met, such as FastAPI going down or when high error rates occur.

Configure Alertmanager
Ensure your email settings are correctly set up in alertmanager.yml. You should update this file with your email and SMTP credentials.

Here’s an example:

yaml
Copy code
receivers:
  - name: 'email-me'
    email_configs:
      - to: 'your-email@example.com'
        from: 'your-email@example.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'your-email@example.com'
        auth_password: 'your-app-password'
        require_tls: true
For Gmail, ensure that you use an App Password instead of your normal password if you have 2-factor authentication enabled.

Start Alertmanager
Once the email is configured, start Alertmanager using the configuration file provided:

bash
Copy code
/path/to/alertmanager --config.file=./monitoring/alertmanager.yml
5. Grafana Setup (Optional)
Grafana is used to visualize metrics scraped by Prometheus. Here’s how you can set it up:

Install Grafana (if you haven’t already).
Start Grafana and access it at http://localhost:3000.
Add Prometheus as a Data Source in Grafana:
URL: http://localhost:9090
Create a new Dashboard or import a pre-built one from Grafana's dashboard marketplace to monitor FastAPI metrics.
Build Visualizations: Use queries like up{job="fastapi"} or rate(http_requests_total[5m]) to visualize FastAPI performance.
6. Using the Setup
Once all the services are running:

Prometheus UI: http://localhost:9090 – Use this to run Prometheus queries and view the metrics being scraped.
FastAPI Metrics: http://localhost:8001/metrics – Metrics are exposed at this endpoint.
Alertmanager: http://localhost:9093 – Alerts will be sent to your email as configured.
Grafana: http://localhost:3000 – Use Grafana to visualize FastAPI metrics.
7. Alerts and Rules
Prometheus will automatically apply the alert rules configured in alerts.yml. Some basic rules include:

HighErrorRate: An alert is triggered if more than 5% of HTTP requests return 500 errors within 5 minutes.
HighLatency: Alerts if the 90th percentile of request duration is greater than 500ms.
InstanceDown: Alerts when the FastAPI instance is down for 5 minutes.
8. Stopping the Services
You can stop the services by pressing Ctrl+C in the terminal where they are running:

Stop Prometheus: Ctrl+C
Stop Alertmanager: Ctrl+C
Stop FastAPI: Ctrl+C
How to Update
To pull the latest changes from the repository and update your local setup, run:

bash
Copy code
git pull origin main
Ensure you restart Prometheus, Alertmanager, and FastAPI after any updates to configuration files.

Troubleshooting
Prometheus is not scraping FastAPI metrics: Ensure FastAPI is running and the /metrics endpoint is available. Verify the Prometheus scrape configuration in prometheus.yml.
Email alerts not working: Check the email configuration in alertmanager.yml. Ensure that you are using an App Password for Gmail.
For any issues, feel free to open a GitHub issue in this repository.

