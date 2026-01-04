from zenml import step
import joblib
from pathlib import Path
from typing_extensions import Annotated

@step
def export_model(
    model,
    version: str
) -> Annotated[str, "model_path"]:

    Path("api/models").mkdir(parents=True, exist_ok=True)
    path = Path(f"api/models/model_{version}.joblib")

    joblib.dump(model, path)
    return str(path)
