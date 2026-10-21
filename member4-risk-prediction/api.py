from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.prediction import (
    predict_historical_incidents,
    get_automatic_historical_counts,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "historical_risk_model.joblib"

app = FastAPI(
    title="Component 4 - Risk Prediction API",
    description="Historical incident prediction backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)



class HistoricalPredictionRequest(BaseModel):
    area: str = Field(min_length=1)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    previous_day_count: int = Field(ge=0)
    previous_7_day_count: int = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Component 4 Risk Prediction API is running",
        "model": "Historical incident count model",
    }


@app.get("/api/historical-counts")
def historical_counts(area: str, date: str):
    if not area.strip():
        raise HTTPException(
            status_code=400,
            detail="Area is required",
        )

    try:
        counts = get_automatic_historical_counts(
            csv_path=BASE_DIR / "Crime_Data_from_2020_to_Present.csv",
            area=area.strip(),
            prediction_date=date,
        )
        return {
            "area": area.strip(),
            "prediction_date": date,
            **counts,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc




@app.post("/api/predict/historical")
def predict_historical(request: HistoricalPredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Historical model is not loaded",
        )

    try:
        result = predict_historical_incidents(
            model=model,
            area=request.area.strip(),
            date=request.date,
            previous_day_count=request.previous_day_count,
            previous_7_day_count=request.previous_7_day_count,
        )

        result.update({
            "previous_day_count": request.previous_day_count,
            "previous_7_day_count": request.previous_7_day_count,
            "count_source": "User-provided inputs",
        })

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
