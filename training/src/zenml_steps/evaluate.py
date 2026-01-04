import mlflow
import numpy as np
from zenml import step
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

@step(experiment_tracker="mlflow_tracker")
def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    exp_name: str,
):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    return metrics
