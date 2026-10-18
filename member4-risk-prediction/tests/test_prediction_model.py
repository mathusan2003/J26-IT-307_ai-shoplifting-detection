
import pytest
from sklearn.linear_model import LogisticRegression

from src.prediction import predict_risk


@pytest.fixture
def trained_model():
    X = [[1, 10], [2, 20], [3, 30], [8, 80], [9, 90], [10, 100]]
    y = [0, 0, 0, 1, 1, 1]

    model = LogisticRegression()
    model.fit(X, y)

    return model


def test_predict_risk_returns_class(trained_model):
    result = predict_risk(trained_model, [[2, 20]])

    assert "predicted_class" in result
    assert result["predicted_class"] in [0, 1]


def test_predict_risk_returns_probabilities(trained_model):
    result = predict_risk(trained_model, [[2, 20]])

    assert "class_probabilities" in result
    assert set(result["class_probabilities"].keys()) == {"0", "1"}

    total = sum(result["class_probabilities"].values())
    assert total == pytest.approx(1.0)


def test_predict_risk_rejects_missing_model():
    with pytest.raises(ValueError, match="trained model"):
        predict_risk(None, [[1, 10]])


def test_predict_risk_rejects_empty_features(trained_model):
    with pytest.raises(ValueError, match="features"):
        predict_risk(trained_model, [])

