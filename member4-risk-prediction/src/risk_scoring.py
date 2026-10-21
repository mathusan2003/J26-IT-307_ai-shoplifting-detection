
def calculate_risk_level(risk_score):
    """Convert a score from 0 to 100 into a risk level."""

    if (
        not isinstance(risk_score, (int, float))
        or isinstance(risk_score, bool)
    ):
        raise ValueError("risk_score must be a number")

    if not 0 <= risk_score <= 100:
        raise ValueError("risk_score must be between 0 and 100")

    if risk_score <= 33:
        return "low"
    elif risk_score <= 66:
        return "medium"
    else:
        return "high"


def calculate_risk(person):
    """Calculate final risk result for one person."""

    if not isinstance(person, dict):
        raise ValueError("person must be a dictionary")

    if "person_id" not in person:
        raise ValueError("person_id is required")

    if "behaviour_score" not in person:
        raise ValueError("behaviour_score is required")

    person_id = person["person_id"]
    risk_score = person["behaviour_score"]

    if not isinstance(person_id, int) or isinstance(person_id, bool):
        raise ValueError("person_id must be an integer")

    risk_level = calculate_risk_level(risk_score)

    return {
        "person_id": person_id,
        "risk_level": risk_level,
        "risk_score": risk_score,
    }
