def breast_cancer_pipeline(model_name: str, exp_name: str, max_iter: int = 1000, n_estimators: int = 100, max_depth: int = 5):
    import mlflow
    import os
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score

    # =========================
    # Data
    # =========================
    df = pd.read_csv("data/raw/breast_cancer.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # MLflow
    # =========================
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=model_name):
        # =========================
        # Model selection
        # =========================
        if model_name == "logreg":
            model = LogisticRegression(max_iter=max_iter)
            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_param("max_iter", max_iter)

        elif model_name == "rf":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            mlflow.log_param("model", "RandomForest")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

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
        model_path = f"artifacts/model_{model_name}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"✅ {model_name} | F1={f1:.4f} | ACC={acc:.4f}")
