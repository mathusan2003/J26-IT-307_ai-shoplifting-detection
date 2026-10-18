
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.prediction import predict_historical_incidents

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "historical_risk_model.joblib"
DATA_PATH = BASE_DIR / "Crime_Data_from_2020_to_Present.csv"
METRICS_PATH = BASE_DIR / "outputs" / "model_metrics.csv"

st.set_page_config(
    page_title="Historical Shoplifting Risk Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("Historical Shoplifting Risk Prediction")
st.write(
    "Component 4: Uses historical shoplifting-related crime records "
    "to estimate next-day incident counts by area."
)

# Load trained model
if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    st.success("Historical prediction model loaded successfully.")
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()


# Load only the columns needed for the dashboard
areas = []

if DATA_PATH.exists():
    try:
        df = pd.read_csv(
            DATA_PATH,
            usecols=["Crm Cd Desc", "AREA NAME"],
            dtype={
                "Crm Cd Desc": "category",
                "AREA NAME": "category",
            },
            on_bad_lines="skip",
        )

        df = df[
            df["Crm Cd Desc"]
            .astype(str)
            .str.contains("shoplift", case=False, na=False)
        ].copy()

        areas = sorted(
            df["AREA NAME"].dropna().astype(str).unique().tolist()
        )

        st.subheader("Dataset Overview")

        col1, col2 = st.columns(2)
        col1.metric("Shoplifting-related records", f"{len(df):,}")
        col2.metric("Areas available", len(areas))

    except Exception as exc:
        st.warning(f"Could not read the dataset: {exc}")
        st.info(
            "You can still try the prediction form by entering an area name."
        )

else:
    st.warning(f"Dataset not found: {DATA_PATH}")


# Prediction form
st.divider()
st.subheader("Predict Next-Day Incident Count")

with st.form("prediction_form"):

    if areas:
        selected_area = st.selectbox("Select area", areas)
    else:
        selected_area = st.text_input("Enter area name", "Central")

    prediction_date = st.date_input(
        "Prediction date",
        value=pd.Timestamp.today().date() + pd.Timedelta(days=1),
    )

    previous_day_count = st.number_input(
        "Incidents on the previous day",
        min_value=0,
        value=0,
        step=1,
    )

    previous_7_day_count = st.number_input(
        "Incidents during the previous 7 days",
        min_value=0,
        value=0,
        step=1,
    )

    submitted = st.form_submit_button("Predict Risk")


if submitted:
    if not selected_area.strip():
        st.error("Please select or enter an area name.")
    else:
        try:
            result = predict_historical_incidents(
                model=model,
                area=selected_area,
                date=prediction_date.strftime("%Y-%m-%d"),
                previous_day_count=previous_day_count,
                previous_7_day_count=previous_7_day_count,
            )

            st.subheader("Prediction Result")

            st.metric(
                "Estimated incidents for the next day",
                result["predicted_incident_count"],
            )

            st.write("Area:", result["area"])
            st.write("Prediction date:", result["prediction_date"])

            st.caption(
                "This is an area-level historical prototype. It is not a "
                "validated prediction for a particular retail store or shelf."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# Model evaluation
if METRICS_PATH.exists():
    st.divider()
    st.subheader("Model Evaluation")

    try:
        metrics = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not load model metrics: {exc}")


st.divider()
st.caption(
    "Research prototype. The current dataset contains historical city-level "
    "crime records. Retail shelf-level risk analysis requires integration "
    "with the Component 3 output."
)
