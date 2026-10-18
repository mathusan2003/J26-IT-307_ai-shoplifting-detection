def calculate_risk_level(risk_score):
    """
    Convert risk score into a risk level.
    """

    if risk_score <= 33:
        return "low"
    elif risk_score <= 66:
        return "medium"
    else:
        return "high"


def calculate_risk(person):
    """
    Calculate final risk result for one person.
    """

    if "person_id" not in person:
        raise ValueError("person_id is required")

    if "behaviour_score" not in person:
        raise ValueError("behaviour_score is required")

    person_id = person["person_id"]
    risk_score = person["behaviour_score"]

    risk_level = calculate_risk_level(risk_score)

    return {
        "person_id": person_id,
        "risk_level": risk_level,
        "risk_score": risk_score
    }