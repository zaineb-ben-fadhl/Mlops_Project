from src.zenml_pipelines.breast_cancer_pipeline import breast_cancer_pipeline

def main() -> None:
    breast_cancer_pipeline.with_options(enable_cache=False)(
        model_name="logreg",
        exp_name="zenml_breast_cancer_baseline",
        version="v3",
    )

if __name__ == "__main__":
    main()
