
import pytest

from src.prediction import validate_prediction_input


def test_valid_prediction_input():
    data = {
        "date": "2026-10-09",
        "time": "14:30",
        "shelf_id": "SHELF_01",
        "previous_incidents": 2
    }

    assert validate_prediction_input(data) is True


def test_missing_shelf_id():
    data = {
        "date": "2026-10-09",
        "time": "14:30",
        "previous_incidents": 2
    }

    with pytest.raises(ValueError, match="shelf_id is required"):
        validate_prediction_input(data)


def test_negative_previous_incidents():
    data = {
        "date": "2026-10-09",
        "time": "14:30",
        "shelf_id": "SHELF_01",
        "previous_incidents": -1
    }

    with pytest.raises(ValueError, match="cannot be negative"):
        validate_prediction_input(data)


from src.prediction import extract_prediction_features


def test_extract_prediction_features():
    data = {
        "date": "2026-10-09",
        "time": "14:30",
        "shelf_id": "SHELF_01",
        "previous_incidents": 2
    }

    result = extract_prediction_features(data)

    assert result["day_of_week"] == "Friday"
    assert result["month"] == 10
    assert result["hour"] == 14
    assert result["shelf_id"] == "SHELF_01"
    assert result["previous_incidents"] == 2


def test_invalid_date_format():
    data = {
        "date": "09-10-2026",
        "time": "14:30",
        "shelf_id": "SHELF_01",
        "previous_incidents": 2
    }

    with pytest.raises(ValueError, match="date must use"):
        extract_prediction_features(data)


from src.prediction import calculate_historical_features


def test_calculate_historical_features():
    daily_incidents = {
        "2026-10-02": 1,
        "2026-10-03": 2,
        "2026-10-05": 1,
        "2026-10-08": 1
    }

    result = calculate_historical_features(
        daily_incidents,
        "2026-10-09"
    )

    assert result["previous_7_day_incidents"] == 5


def test_historical_features_reject_future_dates():
    daily_incidents = {
        "2026-10-10": 2
    }

    with pytest.raises(ValueError, match="before current_date"):
        calculate_historical_features(
            daily_incidents,
            "2026-10-09"
        )
