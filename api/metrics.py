from prometheus_client import Counter, Histogram

# Comptage des requêtes HTTP avec tous les labels utilisés dans le middleware
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["model_version", "endpoint", "status"]
)

# Latence des requêtes HTTP
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["model_version", "endpoint"]
)

# Initialisation par défaut pour éviter crash /metrics avant toute requête
for version in ["v1", "v2", "unknown"]:
    REQUEST_COUNT.labels(model_version=version, endpoint="/", status="200")
    REQUEST_LATENCY.labels(model_version=version, endpoint="/")
