"""
Lancement du pipeline avec MLflow + MinIO (Docker Compose).
"""

import os
from zenml_pipelines.breast_cancer_pipeline import breast_cancer_pipeline

def main() -> None:
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"

    breast_cancer_pipeline(
        model_name="logreg",
        exp_name="zenml_bc_minio",
    )

if __name__ == "__main__":
    main()
