import os
import subprocess
import sys


def set_docker_username():
	if 'YOUR_DOCKER_USERNAME' not in os.environ:
		docker_username = input("Please enter your Docker username: ").strip()
		if not docker_username:
			print("Error: Docker username cannot be empty.")
			sys.exit(1)
		os.environ['YOUR_DOCKER_USERNAME'] = docker_username
	return os.environ['YOUR_DOCKER_USERNAME']


def run_command(command):
	process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
	output, error = process.communicate()
	if process.returncode != 0:
		print(f"Error executing command: {command}")
		print(error.decode('utf-8'))
		sys.exit(1)
	return output.decode('utf-8')


def ask_yes_no(question):
	while True:
		response = input(f"{question} (yes/no): ").lower().strip()
		if response in ['yes', 'y']:
			return True
		elif response in ['no', 'n']:
			return False
		else:
			print("Please answer with 'yes' or 'no'.")


def setup_and_push_docker_image(docker_username):
	if ask_yes_no("Do you want to rebuild and repush the Docker image?"):
		print("Building Docker image...")
		run_command("docker build -t stock-prediction-app .")

		print("Tagging Docker image...")
		run_command(f"docker tag stock-prediction-app:latest {docker_username}/stock-prediction-app:latest")

		print("Logging into Docker Hub...")
		run_command("docker login")

		print("Pushing Docker image...")
		run_command(f"docker push {docker_username}/stock-prediction-app:latest")
		print("Docker image rebuilt and repushed successfully.")
	else:
		print("Skipping Docker image rebuild and repush.")


def deploy_to_kubernetes(docker_username):
	if ask_yes_no("Do you want to reinitialize the Kubernetes deployment?"):
		print("Applying Kubernetes configuration...")
		with open('kubernetes-deployment.yaml', 'r') as file:
			k8s_deployment_config = file.read()

		k8s_deployment_config = k8s_deployment_config.replace('${DOCKER_USERNAME}', docker_username)

		with open('temp_kube_config.yaml', 'w') as file:
			file.write(k8s_deployment_config)

		run_command("kubectl apply -f temp_kube_config.yaml")
		os.remove('temp_kube_config.yaml')
		print("Kubernetes deployment reinitialized successfully.")
	else:
		print("Skipping Kubernetes deployment reinitialization.")


docker_username = set_docker_username()
print(f"Please enter your Docker username, in order to build, push, initialize Docker image for Kubernetes deployment: {docker_username}")

setup_and_push_docker_image(docker_username)
deploy_to_kubernetes(docker_username)

print("Docker setup and Kubernetes deployment completed successfully! Ready to start the app ...")
