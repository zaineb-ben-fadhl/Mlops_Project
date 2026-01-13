import time
from fastapi import APIRouter, HTTPException
from api.v1.schemas import PredictionRequest, PredictionResponse
from api.v1.service import predict
from api.metrics import REQUEST_COUNT, REQUEST_LATENCY

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    start_time = time.time()
    try:
        pred, proba = predict(request.features)

        # Incrémentation des métriques pour v1
        REQUEST_COUNT.labels(
            model_version="v1",
            endpoint="/api/v1/predict",
            status="200"
        ).inc()
        REQUEST_LATENCY.labels(
            model_version="v1",
            endpoint="/api/v1/predict"
        ).observe(time.time() - start_time)

        return PredictionResponse(
            prediction=pred,
            probability=proba
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
