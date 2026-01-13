from src.zenml_pipelines.breast_cancer_pipeline import breast_cancer_pipeline

def main():
    configs = [
        {"model_name": "logreg", "exp_name": "zenml_bc_logreg_500", "version": "v1", "max_iter": 500},
        {"model_name": "logreg", "exp_name": "zenml_bc_logreg_1000", "version": "v2", "max_iter": 1000},
        {"model_name": "rf", "exp_name": "zenml_bc_rf_100", "version": "v1", "n_estimators": 100, "max_depth": 3},
        {"model_name": "rf", "exp_name": "zenml_bc_rf_200", "version": "v2", "n_estimators": 200, "max_depth": 5},
    ]

    for cfg in configs:
        print(f"[ZenML] Lancement pipeline avec paramètres : {cfg}")
        breast_cancer_pipeline(**cfg)

if __name__ == "__main__":
    main()
