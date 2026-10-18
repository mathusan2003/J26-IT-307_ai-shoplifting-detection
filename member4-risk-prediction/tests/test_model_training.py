
import pytest

from src.model_training import (
    validate_training_data,
    split_training_data,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    evaluate_model,
    compare_models,
)


def test_valid_training_data():
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 0]

    assert validate_training_data(X, y) is True


def test_missing_features():
    with pytest.raises(ValueError, match="required"):
        validate_training_data(None, [0, 1])


def test_empty_training_data():
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_training_data([], [])


def test_mismatched_rows():
    X = [[1, 2], [3, 4]]
    y = [0]

    with pytest.raises(ValueError, match="same number of rows"):
        validate_training_data(X, y)


def test_split_training_data():
    X = [[i, i + 1] for i in range(20)]
    y = [i % 2 for i in range(20)]

    X_train, X_test, y_train, y_test = split_training_data(X, y)

    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4


def test_invalid_test_size():
    X = [[1], [2], [3], [4]]
    y = [0, 1, 0, 1]

    with pytest.raises(ValueError, match="test_size"):
        split_training_data(X, y, test_size=1)


def test_train_logistic_regression():
    X_train = [
        [1, 2], [2, 3], [3, 4],
        [6, 7], [7, 8], [8, 9]
    ]
    y_train = [0, 0, 0, 1, 1, 1]

    model = train_logistic_regression(X_train, y_train)

    assert model is not None
    assert model.classes_.tolist() == [0, 1]


def test_train_random_forest():
    X_train = [
        [1, 2], [2, 3], [3, 4],
        [6, 7], [7, 8], [8, 9]
    ]
    y_train = [0, 0, 0, 1, 1, 1]

    model = train_random_forest(X_train, y_train)

    assert model is not None
    assert model.classes_.tolist() == [0, 1]


def test_train_xgboost():
    X_train = [
        [1, 10], [2, 20], [3, 30],
        [4, 40], [5, 50], [6, 60]
    ]
    y_train = [0, 0, 0, 1, 1, 1]

    model = train_xgboost(X_train, y_train)
    predictions = model.predict(X_train)

    assert model is not None
    assert len(predictions) == len(y_train)


def test_evaluate_model():
    X_train = [
        [1, 2], [2, 3], [3, 4],
        [6, 7], [7, 8], [8, 9]
    ]
    y_train = [0, 0, 0, 1, 1, 1]

    X_test = [[1, 3], [7, 9], [2, 4], [8, 10]]
    y_test = [0, 1, 0, 1]

    model = train_logistic_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics.keys()) == {
        "accuracy", "precision", "recall", "f1_score"
    }

    for score in metrics.values():
        assert 0 <= score <= 1


def test_compare_models():
    X_train = [
        [1, 2], [2, 3], [3, 4],
        [6, 7], [7, 8], [8, 9]
    ]
    y_train = [0, 0, 0, 1, 1, 1]

    X_test = [[1, 3], [7, 9], [2, 4], [8, 10]]
    y_test = [0, 1, 0, 1]

    results = compare_models(
        X_train, y_train, X_test, y_test
    )

    assert "Logistic Regression" in results
    assert "Random Forest" in results
    assert "XGBoost" in results

    for metrics in results.values():
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
