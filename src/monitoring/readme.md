Created: MMS

ALARM: ensure they update the **alertmanager.yml** with your email credentials.

Step 1: Ensure You Have Installed the Required Tools
Make sure you have the following installed:

Prometheus
Alertmanager
Grafana
Helm
Kubernetes cluster (running in Docker Desktop)
If these tools are not installed, install them accordingly before proceeding.

Step 2: Start Prometheus with Your prometheus.yml
Run Prometheus:
Since your prometheus.yml is now in the project directory, run Prometheus with the following command:

bash
Copy code
prometheus --config.file=./monitoring/prometheus.yml
This starts Prometheus with the configuration you specified, scraping FastAPI and Kubernetes metrics.

Verify Prometheus:
Visit http://localhost:9090 in your browser. Here, you should see the Prometheus UI. You can check if Prometheus is scraping the targets by going to Status > Targets and verifying that fastapi and kubernetes metrics are being scraped.

Step 3: Start Alertmanager
Next, start Alertmanager to receive alerts defined in your alertmanager.yml file:

bash
Copy code
cd /path/to/your/alertmanager
./alertmanager --config.file=alertmanager.yml
This starts the Alertmanager instance.

Verify Alertmanager:
Visit http://localhost:9093 in your browser to access the Alertmanager UI.

Step 4: Deploy Prometheus Monitoring Stack on Kubernetes
Install Prometheus in Kubernetes:

Run the following command using Helm to install the Prometheus stack in Kubernetes (including the Node Exporter, kube-state-metrics, and cAdvisor):

bash
Copy code
helm install prometheus prometheus-community/kube-prometheus-stack
This deploys Prometheus and its required components to your Kubernetes cluster.

Verify the Kubernetes Stack:

Check the running services in Kubernetes by using:

bash
Copy code
kubectl get svc
Ensure that services like prometheus-kube-prometheus-prometheus, prometheus-kube-prometheus-alertmanager, and prometheus-node-exporter are running.

Step 5: Start Grafana
Run Grafana:

If you have Grafana installed via Helm, run the following command:

bash
Copy code
helm install grafana grafana/grafana
Access Grafana:

After running Grafana, visit it by accessing http://localhost:3000.

Set Up Prometheus Data Source in Grafana:

Go to Configuration > Data Sources > Add New Data Source.
Select Prometheus and set the URL to http://prometheus-service:9090 (or http://localhost:9090 if Prometheus is running locally).
Click Save & Test to verify.
Import Kubernetes Dashboard in Grafana:

Go to + (Add) > Import.
Use an existing Kubernetes dashboard template from Grafana's library. You can use ID: 6417 for a full Kubernetes monitoring dashboard.
Step 6: View Metrics and Alerts
Prometheus Metrics:

You can query Prometheus metrics by visiting http://localhost:9090 and running queries like:

For FastAPI metrics:
bash
Copy code
up{job="fastapi"}
For Kubernetes node metrics:
bash
Copy code
node_memory_MemAvailable_bytes
node_cpu_seconds_total
Alertmanager:

Check if any alerts are firing on http://localhost:9093 (Alertmanager).

Grafana Dashboards:

Use Grafana to create dashboards to visualize the scraped metrics.
You can visualize both FastAPI metrics and Kubernetes cluster metrics on the same dashboard by querying Prometheus.
Step 7: Receive Alerts via Email
Once Alertmanager is properly set up, and you have verified that your alerts are defined correctly in alerts.yml, you will start receiving emails if any of your alert conditions are met.

Make sure you have followed the steps to configure your Gmail account's App Password so that Alertmanager can send emails through it.

Final Checklist:
Prometheus is running and scraping metrics from both FastAPI and Kubernetes.
Alertmanager is running and connected to Prometheus for receiving alerts.
Grafana is running, and you have connected it to Prometheus as a data source.
Alerts are configured to fire based on the alert rules defined in alerts.yml.
