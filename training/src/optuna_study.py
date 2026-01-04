import optuna
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =============================
# CONFIG
# =============================
DATA_PATH = "data/raw/breast_cancer.csv"
EXPERIMENT_NAME = "breast_cancer_optuna"
N_TRIALS = 5

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# =============================
# LOAD DATA
# =============================
data = pd.read_csv(DATA_PATH)

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =============================
# OPTUNA OBJECTIVE
# =============================
def objective(trial):

    C = trial.suggest_float("C", 1e-3, 10.0, log=True)
    max_iter = trial.suggest_int("max_iter", 200, 600)
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
    class_weight = trial.suggest_categorical(
        "class_weight", [None, "balanced"]
    )

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        penalty="l2",
        class_weight=class_weight
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    with mlflow.start_run(nested=True):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # ---- MLflow logs
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("solver", solver)
        mlflow.log_param("class_weight", class_weight)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(pipeline, "model")

    return f1  # 🔥 optimisation sur F1 (meilleur choix)

# =============================
# RUN STUDY
# =============================
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n✅ BEST OPTUNA RESULT")
    print("F1-score:", study.best_value)
    print("Best params:", study.best_params)
