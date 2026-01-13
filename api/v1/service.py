from pathlib import Path
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model_v1.joblib"

_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict(features: list[float]):
    if len(features) != 30:
        raise ValueError("Expected 30 features")

    model = get_model()
    X = np.array(features).reshape(1, -1)

    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0].max())

    # Pour v1, on ne touche pas à la prédiction
    return pred, proba

