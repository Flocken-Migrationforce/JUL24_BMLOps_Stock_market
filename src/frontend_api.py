# frontend_api.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import httpx

app = FastAPI()

templates = Jinja2Templates(directory="HTML")
app.mount("/static", StaticFiles(directory="static"), name="static")

BACKEND_URL = "http://localhost:8001"
DATA_MODEL_URL = "http://localhost:8002"

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
async def train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

@app.post("/train")
async def train_model(request: Request):
    form_data = await request.form()
    symbol = form_data.get("symbol")

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{DATA_MODEL_URL}/train/", json={"symbol": symbol})

    result = response.json()
    return templates.TemplateResponse("train_result.html", {"request": request, "result": result})

@app.get("/predict")
async def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict")
async def predict_stock(request: Request):
    form_data = await request.form()
    symbol = form_data.get("symbol")

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{DATA_MODEL_URL}/predict/", json={"symbol": symbol})

    result = response.json()
    return templates.TemplateResponse("predict_result.html", {"request": request, "result": result})

# Add more routes as needed


if __name__ == '__main__':
	import uvicorn
	uvicorn.run("data_model:app", host="0.0.0.0", port=8000, reload=True)
	# uvicorn backend:app --port 8000