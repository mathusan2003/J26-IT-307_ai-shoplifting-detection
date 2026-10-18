
def validate_training_data(X, y):
    """Validate features and target labels before training."""

    if X is None or y is None:
        raise ValueError("Features X and target y are required")

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Training data cannot be empty")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")

    return True


def split_training_data(X, y, test_size=0.2):
    """Split data into training and testing sets."""

    from sklearn.model_selection import train_test_split

    validate_training_data(X, y)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if len(X) < 2:
        raise ValueError("At least two samples are required")

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )


def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression baseline model."""

    from sklearn.linear_model import LogisticRegression

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data cannot be empty")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    """Train a Random Forest classification model."""

    from sklearn.ensemble import RandomForestClassifier

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data cannot be empty")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_xgboost(X_train, y_train):
    """Train an XGBoost classification model."""

    from xgboost import XGBClassifier

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data cannot be empty")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained classification model."""

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("Testing data cannot be empty")

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test must contain the same number of rows"
        )

    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        )
    }


def compare_models(X_train, y_train, X_test, y_test):
    """Train and compare three classification models."""

    models = {
        "Logistic Regression": train_logistic_regression(
            X_train, y_train
        ),
        "Random Forest": train_random_forest(
            X_train, y_train
        ),
        "XGBoost": train_xgboost(
            X_train, y_train
        )
    }

    results = {}

    for name, model in models.items():
        results[name] = evaluate_model(
            model, X_test, y_test
        )

    return results
