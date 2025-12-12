# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="Wine Quality Prediction API")

# 1. Load the Model
# PASTE YOUR FULL PATH BELOW inside the quotes
# Example: model_path = "./mlruns/0/a1b2c3.../artifacts/model"
# Change the model path to simply:
model_path = "./model"

try:
    model = mlflow.sklearn.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# 2. Define the Input Data Structure (Schema)
# This acts as a contract. If the user sends bad data, the API rejects it.
class WineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


# 3. Define the Prediction Endpoint
@app.post("/predict")
def predict(data: WineData):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Create the DataFrame from the input JSON
    input_df = pd.DataFrame([data.dict()])

    # 2. FIX: Rename columns to match the training data format
    # Replace underscores (_) with spaces ( )
    input_df.columns = [col.replace("_", " ") for col in input_df.columns]

    # 3. Make prediction
    prediction = model.predict(input_df)

    return {"predicted_quality": float(prediction[0])}