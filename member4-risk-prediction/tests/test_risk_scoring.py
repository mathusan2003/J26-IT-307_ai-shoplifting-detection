
import pytest

from src.risk_scoring import calculate_risk


def test_low_risk():
    result = calculate_risk({
        "person_id": 1,
        "behaviour_score": 20,
    })
    assert result["risk_level"] == "low"


def test_medium_risk():
    result = calculate_risk({
        "person_id": 2,
        "behaviour_score": 45,
    })
    assert result["risk_level"] == "medium"


def test_high_risk():
    result = calculate_risk({
        "person_id": 3,
        "behaviour_score": 80,
    })
    assert result["risk_level"] == "high"


def test_missing_person_id():
    with pytest.raises(ValueError):
        calculate_risk({"behaviour_score": 45})


def test_missing_behaviour_score():
    with pytest.raises(ValueError):
        calculate_risk({"person_id": 1})


def test_invalid_score_above_100():
    with pytest.raises(ValueError):
        calculate_risk({
            "person_id": 1,
            "behaviour_score": 120,
        })


def test_invalid_negative_score():
    with pytest.raises(ValueError):
        calculate_risk({
            "person_id": 1,
            "behaviour_score": -5,
        })


def test_invalid_person_id_type():
    with pytest.raises(ValueError):
        calculate_risk({
            "person_id": "person-1",
            "behaviour_score": 45,
        })
