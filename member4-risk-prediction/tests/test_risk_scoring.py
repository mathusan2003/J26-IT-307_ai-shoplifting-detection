import pytest

from src.risk_scoring import calculate_risk


def test_low_risk():
    person = {
        "person_id": 1,
        "behaviour_score": 20
    }

    result = calculate_risk(person)

    assert result["person_id"] == 1
    assert result["risk_score"] == 20
    assert result["risk_level"] == "low"


def test_medium_risk():
    person = {
        "person_id": 2,
        "behaviour_score": 45
    }

    result = calculate_risk(person)

    assert result["risk_level"] == "medium"


def test_high_risk():
    person = {
        "person_id": 3,
        "behaviour_score": 80
    }

    result = calculate_risk(person)

    assert result["risk_level"] == "high"


def test_missing_person_id():
    person = {
        "behaviour_score": 45
    }

    with pytest.raises(ValueError):
        calculate_risk(person)


def test_missing_behaviour_score():
    person = {
        "person_id": 1
    }

    with pytest.raises(ValueError):
        calculate_risk(person)