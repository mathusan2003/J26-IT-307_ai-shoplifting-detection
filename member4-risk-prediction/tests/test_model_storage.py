
import pytest
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from src.model_storage import save_model, load_model


def test_save_and_load_model(tmp_path):
    X = [[1], [2], [3], [4]]
    y = [0, 0, 1, 1]

    model = LogisticRegression()
    model.fit(X, y)

    file_path = tmp_path / "models" / "test_model.joblib"

    saved_path = save_model(model, file_path)

    assert Path(saved_path).exists()

    loaded_model = load_model(file_path)

    assert loaded_model is not None
    assert loaded_model.predict(X).tolist() == model.predict(X).tolist()


def test_save_missing_model(tmp_path):
    file_path = tmp_path / "models" / "missing.joblib"

    with pytest.raises(ValueError, match="Model is required"):
        save_model(None, file_path)


def test_load_missing_model(tmp_path):
    file_path = tmp_path / "missing_model.joblib"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model(file_path)
