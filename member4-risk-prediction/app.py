
from pathlib import Path
import json
import tempfile

import joblib
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from src.prediction import predict_historical_incidents
from src.video_features import extract_features, FEATURE_NAMES


# --------------------------------------------------
# Paths and page setup
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

HISTORICAL_MODEL_PATH = (
    BASE_DIR / "models" / "historical_risk_model.joblib"
)
VIDEO_MODEL_PATH = BASE_DIR / "models" / "video_risk_model.joblib"

DATA_PATH = BASE_DIR / "Crime_Data_from_2020_to_Present.csv"
METRICS_PATH = BASE_DIR / "outputs" / "model_metrics.csv"
VIDEO_METRICS_PATH = BASE_DIR / "outputs" / "video_model_metrics.json"

st.set_page_config(
    page_title="Component 4 - Risk Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("Component 4 — Risk Prediction Dashboard")
st.write(
    "Historical incident estimates and experimental video-based "
    "predictions for the AI shoplifting detection research project."
)


# --------------------------------------------------
# Cached model loaders
# --------------------------------------------------

@st.cache_resource
def load_historical_model():
    return joblib.load(HISTORICAL_MODEL_PATH)


@st.cache_resource
def load_video_bundle():
    return joblib.load(VIDEO_MODEL_PATH)


@st.cache_resource
def load_yolo_detector():
    return YOLO("yolo11n.pt")


# --------------------------------------------------
# Section 1: Historical prediction
# --------------------------------------------------

st.header("1. Historical Incident Prediction")

historical_model = None
areas = []

if HISTORICAL_MODEL_PATH.exists():
    try:
        historical_model = load_historical_model()
        st.success("Historical prediction model loaded.")
    except Exception as exc:
        st.error(f"Could not load historical model: {exc}")
else:
    st.error(f"Historical model not found: {HISTORICAL_MODEL_PATH}")


# Load historical dataset
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

        col1, col2 = st.columns(2)
        col1.metric("Shoplifting-related records", f"{len(df):,}")
        col2.metric("Areas available", len(areas))

    except Exception as exc:
        st.warning(f"Could not read historical dataset: {exc}")
else:
    st.warning(f"Historical dataset not found: {DATA_PATH}")


# Automatic date
today = pd.Timestamp.now().date()
tomorrow = today + pd.Timedelta(days=1)

st.info(f"Today's date: {today:%Y-%m-%d}")

with st.form("historical_prediction_form"):
    if areas:
        selected_area = st.selectbox("Select area", areas)
    else:
        selected_area = st.text_input("Enter area name", "Central")

    prediction_date = st.date_input(
        "Prediction date",
        value=tomorrow,
        min_value=tomorrow,
    )


    historical_submitted = st.form_submit_button(
        "Predict Historical Incidents"
    )


if historical_submitted:
    if not selected_area.strip():
        st.error("Please select or enter an area.")
    elif historical_model is None:
        st.error("Historical model is unavailable.")
    else:
        try:
            result = predict_historical_incidents(
                model=historical_model,
                area=selected_area,
                date=prediction_date.strftime("%Y-%m-%d"),
                previous_day_count=int(previous_day_count),
                previous_7_day_count=int(previous_7_day_count),
            )

            st.subheader("Historical Prediction Result")

            st.metric(
                "Estimated incidents for the selected date",
                f"{result['predicted_incident_count']:.2f}",
            )

            col1, col2 = st.columns(2)
            col1.write(f"**Area:** {result['area']}")
            col2.write(
                f"**Prediction date:** {result['prediction_date']}"
            )

            st.info(
                "This is an estimated incident count based on historical "
                "data. It is not the probability that an incident will occur."
            )

            st.caption(
                "This model uses historical area-level crime records. "
                "It is not yet a validated prediction for an individual "
                "retail store or shelf."
            )

        except Exception as exc:
            st.error(f"Historical prediction failed: {exc}")


# Historical model evaluation
if METRICS_PATH.exists():
    st.subheader("Historical Model Evaluation")

    try:
        metrics_df = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics_df, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not load historical metrics: {exc}")


# --------------------------------------------------
# Section 2: Video prediction
# --------------------------------------------------

st.divider()
st.header("2. Video-Based Prediction")

st.warning(
    "Experimental baseline: this model uses person-detection statistics. "
    "It does not directly identify theft actions and can make mistakes."
)

if not VIDEO_MODEL_PATH.exists():
    st.error(f"Video model not found: {VIDEO_MODEL_PATH}")
else:
    uploaded_video = st.file_uploader(
        "Upload a short MP4 video",
        type=["mp4"],
        help="Upload a video you have permission to analyse.",
        key="video_upload",
    )

    if uploaded_video is not None:
        if st.button("Analyse Uploaded Video", type="primary"):
            temp_path = None

            try:
                with st.spinner(
                    "Loading models and analysing video frames..."
                ):
                    video_bundle = load_video_bundle()
                    video_model = video_bundle["model"]
                    saved_features = video_bundle.get(
                        "features", FEATURE_NAMES
                    )

                    if list(saved_features) != list(FEATURE_NAMES):
                        raise ValueError(
                            "Saved model features do not match "
                            "src/video_features.py. Check the model."
                        )

                    detector = load_yolo_detector()

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".mp4",
                    ) as temp_file:
                        temp_file.write(uploaded_video.getvalue())
                        temp_path = temp_file.name

                    features = extract_features(temp_path, detector)

                    feature_frame = pd.DataFrame(
                        [features],
                        columns=FEATURE_NAMES,
                    )

                    prediction = int(
                        video_model.predict(feature_frame)[0]
                    )

                    probabilities = video_model.predict_proba(
                        feature_frame
                    )[0]

                    probability_by_class = {
                        int(label): float(probability)
                        for label, probability in zip(
                            video_model.classes_,
                            probabilities,
                        )
                    }

                    normal_score = probability_by_class.get(0, 0.0)
                    shoplifting_score = probability_by_class.get(1, 0.0)

                st.subheader("Video Prediction Result")

                if prediction == 1:
                    st.metric("Model output", "Shoplifting label")
                else:
                    st.metric("Model output", "Normal label")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Normal-class model score",
                    f"{normal_score * 100:.1f}%",
                )

                col2.metric(
                    "Shoplifting-class model score",
                    f"{shoplifting_score * 100:.1f}%",
                )

                st.caption(
                    "These are model class scores, not verified "
                    "probabilities that theft occurred."
                )

                st.subheader("Extracted Person-Detection Features")
                st.dataframe(
                    pd.DataFrame([features]),
                    use_container_width=True,
                )

                st.info(
                    "Do not use this prediction alone to accuse anyone. "
                    "The model is experimental and requires further "
                    "behaviour-based features and validation."
                )

            except Exception as exc:
                st.error(f"Video analysis failed: {exc}")

            finally:
                if temp_path and Path(temp_path).exists():
                    Path(temp_path).unlink(missing_ok=True)


# Video model evaluation
if VIDEO_METRICS_PATH.exists():
    st.subheader("Video Model Evaluation")

    try:
        with open(
            VIDEO_METRICS_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            video_metrics = json.load(file)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Training videos",
            video_metrics.get("train_videos", 0),
        )

        col2.metric(
            "Test videos",
            video_metrics.get("test_videos", 0),
        )

        col3.metric(
            "Macro F1",
            f"{video_metrics.get('f1_macro', 0.0):.3f}",
        )

        report = video_metrics.get("classification_report", {})

        if report:
            st.write("Classification report")
            st.dataframe(
                pd.DataFrame(report).transpose(),
                use_container_width=True,
            )

        matrix = video_metrics.get("confusion_matrix")

        if matrix:
            st.write(
                "Confusion matrix "
                "(rows = actual, columns = predicted)"
            )

            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=["Actual normal", "Actual shoplifting"],
                    columns=[
                        "Predicted normal",
                        "Predicted shoplifting",
                    ],
                ),
                use_container_width=True,
            )

        st.caption(
            video_metrics.get(
                "note",
                "Experimental video classification model.",
            )
        )

    except Exception as exc:
        st.warning(f"Could not load video evaluation: {exc}")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Research prototype. Historical crime records and video "
    "person-detection features are separate sources and are not combined "
    "into one risk score. Automatic live CCTV processing and validated "
    "store-level risk prediction require further integration."
)
