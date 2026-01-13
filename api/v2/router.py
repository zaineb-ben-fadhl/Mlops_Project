import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.v2.service import predict
from api.metrics import REQUEST_COUNT, REQUEST_LATENCY

class PredictionRequest(BaseModel):
    features: list[float]

router = APIRouter()

@router.post("/predict")
def predict_endpoint(request: PredictionRequest):
    start_time = time.time()
    try:
        pred, proba = predict(request.features)

        # Incrémentation des métriques pour v2
        REQUEST_COUNT.labels(
            model_version="v2",
            endpoint="/api/v2/predict",
            status="200"
        ).inc()
        REQUEST_LATENCY.labels(
            model_version="v2",
            endpoint="/api/v2/predict"
        ).observe(time.time() - start_time)

        return {
            "prediction": pred,
            "probability": proba,
            "model_version": "v2"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
