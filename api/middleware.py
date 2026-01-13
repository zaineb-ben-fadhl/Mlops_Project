import time
from fastapi import Request
from api.metrics import REQUEST_COUNT, REQUEST_LATENCY

async def monitoring_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time
    path = request.url.path

    # Détection de la version
    if path.startswith("/api/v1"):
        model_version = "v1"
    elif path.startswith("/api/v2"):
        model_version = "v2"
    else:
        model_version = "unknown"

    # Comptage
    REQUEST_COUNT.labels(
        model_version=model_version,
        endpoint=path,
        status=str(response.status_code)
    ).inc()

    # Latence
    REQUEST_LATENCY.labels(
        model_version=model_version,
        endpoint=path
    ).observe(latency)

    return response
