from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from api.v1.router import router as v1_router
from api.v2.router import router as v2_router
from api.middleware import monitoring_middleware

app = FastAPI(
    title="Breast Cancer Prediction API",
    version="1.0.0"
)

app.middleware("http")(monitoring_middleware)

@app.get("/")
def root():
    return {"status": "API is running"}

# 🔹 METRICS ENDPOINT
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

app.include_router(v1_router, prefix="/api/v1", tags=["v1"])
app.include_router(v2_router, prefix="/api/v2", tags=["v2"])
