
def validate_prediction_input(data):
    """
    Validate the input required for future risk prediction.
    """

    required_fields = [
        "date",
        "time",
        "shelf_id",
        "previous_incidents"
    ]

    for field in required_fields:
        if field not in data:
            raise ValueError(f"{field} is required")

    if not isinstance(data["shelf_id"], str) or not data["shelf_id"].strip():
        raise ValueError("shelf_id must be a non-empty string")

    if data["previous_incidents"] < 0:
        raise ValueError("previous_incidents cannot be negative")

    return True


def extract_prediction_features(data):
    """
    Extract basic features for future shoplifting risk prediction.
    """

    from datetime import datetime

    validate_prediction_input(data)

    try:
        date_value = datetime.strptime(data["date"], "%Y-%m-%d")
        time_value = datetime.strptime(data["time"], "%H:%M")
    except ValueError as exc:
        raise ValueError(
            "date must use YYYY-MM-DD and time must use HH:MM format"
        ) from exc

    return {
        "day_of_week": date_value.strftime("%A"),
        "month": date_value.month,
        "hour": time_value.hour,
        "shelf_id": data["shelf_id"].strip(),
        "previous_incidents": data["previous_incidents"]
    }


def calculate_historical_features(daily_incidents, current_date):
    """
    Calculate historical incident features for a shelf.

    daily_incidents: dictionary mapping dates to incident counts.
    Example: {"2026-10-01": 2, "2026-10-02": 1}
    current_date: date to predict, in YYYY-MM-DD format.
    """

    from datetime import datetime, timedelta

    try:
        target_date = datetime.strptime(current_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("current_date must use YYYY-MM-DD format") from exc

    # Validate and normalize historical dates and counts.
    normalized = {}

    for date_text, count in daily_incidents.items():
        try:
            incident_date = datetime.strptime(date_text, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(
                f"Invalid historical date: {date_text}"
            ) from exc

        if incident_date >= target_date:
            raise ValueError("Historical dates must be before current_date")

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Incident counts must be non-negative integers")

        normalized[incident_date] = count

    last_7_days = [
        target_date - timedelta(days=offset)
        for offset in range(1, 8)
    ]

    previous_7_day_count = sum(
        normalized.get(day, 0) for day in last_7_days
    )

    return {
        "previous_7_day_incidents": previous_7_day_count
    }


def predict_risk(model, features):
    """
    Predict the risk class and probability using a trained model.

    The features must already be preprocessed in the same way
    as the data used to train the model.
    """

    if model is None:
        raise ValueError("A trained model is required")

    if features is None or len(features) == 0:
        raise ValueError("Prediction features are required")

    prediction = model.predict(features)
    result = {
        "predicted_class": prediction[0]
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        result["class_probabilities"] = {
            str(label): float(probability)
            for label, probability in zip(
                model.classes_, probabilities
            )
        }

    return result

def predict_historical_incidents(model, area, date, previous_day_count, previous_7_day_count):
    """
    Predict next-day shoplifting-related incident count
    using the trained historical area-level model.
    """

    import pandas as pd
    from datetime import datetime

    if model is None:
        raise ValueError("A trained model is required")

    if not isinstance(area, str) or not area.strip():
        raise ValueError("area must be a non-empty string")

    try:
        date_value = datetime.strptime(date, "%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise ValueError("date must use YYYY-MM-DD format") from exc

    for name, value in [
        ("previous_day_count", previous_day_count),
        ("previous_7_day_count", previous_7_day_count),
    ]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a non-negative number")

    features = pd.DataFrame([{
        "AREA NAME": area.strip(),
        "DAY_OF_WEEK": date_value.weekday(),
        "MONTH": date_value.month,
        "DAY_OF_MONTH": date_value.day,
        "LAG_1": previous_day_count,
        "LAG_7": previous_7_day_count,
    }])

    predicted_count = float(model.predict(features)[0])
    predicted_count = max(0.0, predicted_count)

    return {
        "area": area.strip(),
        "prediction_date": date,
        "predicted_incident_count": round(predicted_count, 2),
        "prediction_type": "Next-day historical incident count",
    }


def get_automatic_historical_counts(csv_path, area, prediction_date):
    """
    Calculate previous-day and previous-7-day shoplifting counts
    automatically from the historical crime dataset.
    """

    import pandas as pd
    from datetime import datetime, timedelta

    target_date = datetime.strptime(
        prediction_date, "%Y-%m-%d"
    ).date()

    start_date = target_date - timedelta(days=7)
    previous_day = target_date - timedelta(days=1)

    df = pd.read_csv(
        csv_path,
        usecols=["DATE OCC", "AREA NAME", "Crm Cd Desc"]
    )

    # Keep shoplifting-related records for the selected area.
    df = df[
        (df["AREA NAME"].str.strip().str.lower() == area.strip().lower())
        & (df["Crm Cd Desc"].str.contains("SHOPLIFT", case=False, na=False))
    ].copy()

    df["DATE OCC"] = pd.to_datetime(
        df["DATE OCC"], errors="coerce"
    )

    df = df.dropna(subset=["DATE OCC"])

    dates = df["DATE OCC"].dt.date

    previous_day_count = int((dates == previous_day).sum())

    previous_7_day_count = int(
        ((dates >= start_date) & (dates < target_date)).sum()
    )

    return {
        "previous_day_count": previous_day_count,
        "previous_7_day_count": previous_7_day_count,
        "count_source": "Historical crime dataset"
    }
