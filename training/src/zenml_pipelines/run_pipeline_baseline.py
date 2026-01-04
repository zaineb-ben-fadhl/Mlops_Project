"""
Script de lancement du pipeline ZenML en mode baseline.
L'appel du pipeline déclenche directement l'exécution.
"""

from src.zenml_pipelines.breast_cancer_pipeline import breast_cancer_pipeline


def main() -> None:
    breast_cancer_pipeline(
        model_name="logreg",
        exp_name="zenml_breast_cancer_baseline",
        version="v1",   # ✅ AJOUT OBLIGATOIRE
    )

if __name__ == "__main__":
    main()
