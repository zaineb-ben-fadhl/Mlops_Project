from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "API is running"

def test_predict_v1():
    response = client.post(
        "/api/v1/predict",
        json={
            "features": [
                20.57,17.77,132.9,1326.0,0.08474,
                0.07864,0.0869,0.07017,0.05433,
                0.4564,1.075,3.425,48.55,0.005903,
                0.03731,0.0438,0.01241,0.01619,0.0034,
                24.99,23.41,158.8,1956.0,0.1238,
                0.1866,0.2416,0.186,0.275,0.08902,0.05
            ]
        }
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
