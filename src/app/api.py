# pip install fastapi uvicorn sqlalchemy alembic databases python-jose[cryptography] passlib[bcrypt] requests pandas numpy keras tensorflow python-dotenv


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import get_daily_stock_prices
from app.model import train_model, load_model_and_predict
import os

# List of supported symbols
SUPPORTED_SYMBOLS = ["AAPL", "GOOGL", "EUR/USD", "GOLD"]
app = FastAPI()

class StockRequest(BaseModel):
    symbol: str 