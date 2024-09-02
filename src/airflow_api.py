from fastapi import FastAPI
# Import necessary Airflow modules

app = FastAPI()

from fastapi import FastAPI
from airflow.api.client.local_client import Client

app = FastAPI()
client = Client()

@app.post("/trigger_dag/")
async def trigger_dag(dag_id: str):
    try:
        client.trigger_dag(dag_id=dag_id)
        return {"message": f"DAG {dag_id} triggered successfully"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("airflow_api:app", host="0.0.0.0", port=8001, reload=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("airflow_api:app", host="0.0.0.0", port=8003, reload=True)