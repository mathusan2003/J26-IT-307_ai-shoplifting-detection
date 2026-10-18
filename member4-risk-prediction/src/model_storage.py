
from pathlib import Path
import joblib


def save_model(model, file_path):
    """Save a trained machine learning model."""

    if model is None:
        raise ValueError("Model is required")

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

    return str(path)


def load_model(file_path):
    """Load a previously saved machine learning model."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {file_path}"
        )

    return joblib.load(path)
