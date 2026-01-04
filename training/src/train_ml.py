import os
import joblib
import mlflow
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# =========================
# Arguments
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf"])
args = parser.parse_args()

MODEL_TYPE = args.model

# =========================
# Data
# =========================
DATA_PATH = "data/raw/breast_cancer.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MLflow
# =========================
mlflow.set_experiment("breast_cancer_comparison")

with mlflow.start_run(run_name=MODEL_TYPE):

    # =========================
    # Model selection
    # =========================
    if MODEL_TYPE == "logreg":
        model = LogisticRegression(max_iter=1000)
        mlflow.log_param("model", "LogisticRegression")

    elif MODEL_TYPE == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42
        )
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 5)

    # =========================
    # Training
    # =========================
    model.fit(X_train, y_train)

    # =========================
    # Evaluation
    # =========================
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("f1", f1)
    mlflow.log_metric("accuracy", acc)

    # =========================
    # Save model
    # =========================
    os.makedirs("artifacts", exist_ok=True)
    model_path = f"artifacts/model_{MODEL_TYPE}.joblib"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    print(f"✅ {MODEL_TYPE} | F1={f1:.4f} | ACC={acc:.4f}")
