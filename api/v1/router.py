from fastapi import APIRouter, HTTPException
from api.v1.schemas import PredictionRequest, PredictionResponse
from api.v1.service import predict

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    try:
        pred, proba = predict(request.features)
        return PredictionResponse(
            prediction=pred,
            probability=proba
        )
    except ValueError as e:
        # Erreur utilisateur → 400 Bad Request
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
