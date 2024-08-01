
# This file contains the logic for interacting with Redis as a message queue.

# app/message_queue.py
from fastapi import FastAPI
import redis
import json

message_queue_app = FastAPI()

redis_client = redis.Redis(host='localhost', port=6379, db=0)

@message_queue_app.post("/enqueue-task/")
async def enqueue_task(task_name: str, task_data: dict):
    task = {
        "name": task_name,
        "data": task_data
    }
    redis_client.lpush("task_queue", json.dumps(task))
    return {"message": "Task enqueued successfully"}

@message_queue_app.get("/dequeue-task/")
async def dequeue_task():
    task = redis_client.rpop("task_queue")
    if task:
        return json.loads(task)
    return {"message": "No tasks in queue"}