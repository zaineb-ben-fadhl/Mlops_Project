import mlflow
from zenml import step
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

@step(experiment_tracker="mlflow_tracker")
def train_model(
    X_train,
    y_train,
    model_name: str,
    exp_name: str,
):
    """
    Entraîne un modèle ML et log dans MLflow.
    """
    if model_name == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
    else:
        raise ValueError("Model not supported")

    model.fit(X_train, y_train)

    mlflow.log_param("model_name", model_name)
    mlflow.log_param("experiment_name", exp_name)
    mlflow.sklearn.log_model(model, "model")

    return model
