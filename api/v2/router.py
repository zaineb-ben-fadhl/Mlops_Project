from fastapi import APIRouter
from pydantic import BaseModel
from api.v2.service import predict

from pathlib import Path

class PredictionRequest(BaseModel):
    features: list[float]

router = APIRouter()

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model_v2.joblib"

@router.post("/predict")
def predict_endpoint(data: PredictionRequest):
    pred, proba = predict(data.features)
    return {
        "prediction": pred,
        "probability": proba,
        "model_version": "v2"
    }
