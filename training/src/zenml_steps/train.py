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
    max_iter: int = 1000,       # pour LogisticRegression
    n_estimators: int = 100,    # pour RandomForest
    max_depth: int = 5           # pour RandomForest
):
    """
    Entraîne un modèle ML et log dans MLflow.
    """
    if model_name == "logreg":
        model = LogisticRegression(max_iter=max_iter)
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
    else:
        raise ValueError("Model not supported")

    model.fit(X_train, y_train)

    mlflow.log_param("model_name", model_name)
    mlflow.log_param("experiment_name", exp_name)
    if model_name == "logreg":
        mlflow.log_param("max_iter", max_iter)
    elif model_name == "rf":
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

    mlflow.sklearn.log_model(model, "model")

    return model
