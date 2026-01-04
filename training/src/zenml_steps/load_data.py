from zenml import step
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated

@step
def load_data() -> tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
    data = pd.read_csv("data/raw/breast_cancer.csv")

    X = data.drop("target", axis=1).values
    y = data["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
