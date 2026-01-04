from api.v1.service import predict as predict_v1

def predict(features: list[float]):
    pred, proba = predict_v1(features)

    # Exemple logique v2 (seuil différent)
    pred_v2 = 1 if proba > 0.7 else 0

    return pred_v2, proba
