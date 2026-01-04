from zenml import pipeline
from src.zenml_steps.load_data import load_data
from src.zenml_steps.train import train_model
from src.zenml_steps.evaluate import evaluate_model
from src.zenml_steps.export_model import export_model

@pipeline
def breast_cancer_pipeline(
    model_name: str,
    exp_name: str,
    version: str,
):
    X_train, X_test, y_train, y_test = load_data()

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name=model_name,
        exp_name=exp_name,
        
    )

    evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        exp_name=exp_name,
    )

    export_model(
        model=model,
        version=version,
    )
