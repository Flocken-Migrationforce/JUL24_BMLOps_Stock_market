# app/docker_tasks.py
from fastapi import FastAPI
import docker

docker_tasks_app = FastAPI()

client = docker.from_env()

@docker_tasks_app.post("/run-container/")
async def run_container(image: str, command: str):
    try:
        container = client.containers.run(image, command, detach=True)
        return {"container_id": container.id, "status": "running"}
    except docker.errors.DockerException as e:
        return {"error": str(e)}

@docker_tasks_app.get("/container-status/{container_id}")
async def container_status(container_id: str):
    try:
        container = client.containers.get(container_id)
        return {"container_id": container_id, "status": container.status}
    except docker.errors.NotFound:
        return {"error": "Container not found"}