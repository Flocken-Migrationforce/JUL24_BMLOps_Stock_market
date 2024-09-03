# pip install fastapi uvicorn sqlalchemy alembic databases python-jose[cryptography] passlib[bcrypt] requests pandas numpy keras tensorflow python-dotenv


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import get_daily_stock_prices
from app.model import train_model, load_model_and_predict
import os


app = FastAPI()

class StockRequest(BaseModel):
    symbol: str 