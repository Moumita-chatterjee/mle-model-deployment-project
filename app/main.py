from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


MODEL_PATH = Path("artifacts/model.joblib")


class PredictionRequest(BaseModel):
    trip_distance: float
    passenger_count: float
    PULocationID: int
    DOLocationID: int
    pickup_hour: int


#model = joblib.load(MODEL_PATH)
model = None

@app.get("/")
def root():
    return {"message": "Taxi duration prediction API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    return {
        "message": "Prediction endpoint is working",
        "input": request.model_dump(),
    }