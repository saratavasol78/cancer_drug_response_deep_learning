from fastapi import FastAPI
from pydantic import BaseModel
import sys
sys.path.append("src")
from predict import predict

app = FastAPI(
    title="Cancer Drug Response Prediction API",
    description="Research/educational API. Not for clinical decisions.",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    features: dict

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Cancer drug response model API",
        "warning": "Research/education only; not clinical decision support."
    }

@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    value = predict(request.features)
    return {
        "predicted_response": value,
        "warning": "Research/education only; not clinical decision support."
    }
