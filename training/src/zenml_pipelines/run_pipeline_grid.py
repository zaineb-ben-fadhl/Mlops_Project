"""
Script de lancement d'une grille de runs du pipeline ZenML.
Chaque appel déclenche un nouveau run.
"""

from src.zenml_pipelines.breast_cancer_pipeline import breast_cancer_pipeline

def main() -> None:

    configs = [
        {"model_name": "logreg", "exp_name": "zenml_bc_logreg"},
        {"model_name": "rf", "exp_name": "zenml_bc_rf"},
    ]

    for cfg in configs:
        print(f"[ZenML] Lancement pipeline avec paramètres : {cfg}")
        breast_cancer_pipeline(**cfg)

if __name__ == "__main__":
    main()
