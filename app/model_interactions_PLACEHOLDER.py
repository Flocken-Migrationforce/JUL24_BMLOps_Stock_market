# PLACEHOLDER, to be filled with Mehdi's model re-training and pulling values
# Version 0.1
# Fabian
# 2408011234


# app/model.py
from fastapi import FastAPI, HTTPException
import joblib
import os

model_app = FastAPI()

MODEL_PATH = "models/trained_model.pkl"


@model_app.post("/train-model/")
async def train_model():
	# Placeholder for model training logic
	# Replace this with your actual model training code
	model = "trained_model"  # Replace with actual model object
	os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
	joblib.dump(model, MODEL_PATH)
	return {"message": "Model trained successfully"}


@model_app.get("/predict/")
async def predict(data: dict):
	if not os.path.exists(MODEL_PATH):
		raise HTTPException(status_code=404, detail="Model not found. Train the model first.")

	model = joblib.load(MODEL_PATH)
	# Placeholder for prediction logic
	# Replace this with your actual prediction code
	prediction = "prediction_result"  # Replace with actual prediction logic
	return {"prediction": prediction}