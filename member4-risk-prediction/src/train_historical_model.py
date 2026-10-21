import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Crime_Data_from_2020_to_Present.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def prepare_data():
    columns = [
        "DATE OCC",
        "AREA NAME",
        "Crm Cd Desc",
    ]

    df = pd.read_csv(DATA_PATH, usecols=columns)

    df = df[
        df["Crm Cd Desc"].str.contains(
            "SHOPLIFTING", case=False, na=False
        )
    ].copy()

    df["DATE OCC"] = pd.to_datetime(
        df["DATE OCC"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )

    df = df.dropna(subset=["DATE OCC", "AREA NAME"])
    df["DATE"] = df["DATE OCC"].dt.normalize()

    daily = (
        df.groupby(["DATE", "AREA NAME"])
        .size()
        .rename("INCIDENT_COUNT")
        .reset_index()
    )

    all_dates = pd.date_range(
        daily["DATE"].min(),
        daily["DATE"].max(),
        freq="D",
    )
    all_areas = sorted(daily["AREA NAME"].unique())

    complete_index = pd.MultiIndex.from_product(
        [all_dates, all_areas],
        names=["DATE", "AREA NAME"],
    )

    daily = (
        daily.set_index(["DATE", "AREA NAME"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    daily = daily.sort_values(["AREA NAME", "DATE"])
    daily["TARGET_NEXT_DAY"] = daily.groupby("AREA NAME")["INCIDENT_COUNT"].shift(-1)

    target_dates = daily["DATE"] + pd.Timedelta(days=1)

    daily["DAY_OF_WEEK"] = target_dates.dt.dayofweek
    daily["MONTH"] = target_dates.dt.month
    daily["DAY_OF_MONTH"] = target_dates.dt.day

    daily["LAG_1"] = daily["INCIDENT_COUNT"]

    daily["LAG_7"] = (
    daily.groupby("AREA NAME")["INCIDENT_COUNT"]
    .transform(lambda s: s.rolling(7).sum())
)

    daily = daily.dropna(subset=["TARGET_NEXT_DAY", "LAG_7"])

    return daily


def main():
    daily = prepare_data()

    features = [
        "AREA NAME",
        "DAY_OF_WEEK",
        "MONTH",
        "DAY_OF_MONTH",
        "LAG_1",
        "LAG_7",
    ]

    target = "TARGET_NEXT_DAY"

    # Chronological split: earlier dates for training, later dates for testing.
    split_date = daily["DATE"].quantile(0.8)

    train = daily[daily["DATE"] < split_date].copy()
    test = daily[daily["DATE"] >= split_date].copy()

    if train.empty or test.empty:
        raise ValueError("Not enough data for chronological train/test split.")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "area",
                OneHotEncoder(handle_unknown="ignore"),
                ["AREA NAME"],
            )
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
               RandomForestRegressor(
                       n_estimators=20,
                       max_depth=10,
                      random_state=42,
                      n_jobs=1,
                      min_samples_leaf=5,
                                         )
            ),
        ]
    )

    model.fit(train[features], train[target])

    predictions = model.predict(test[features]).clip(min=0)

    mae = mean_absolute_error(test[target], predictions)
    rmse = mean_squared_error(
        test[target], predictions
    ) ** 0.5

        # Baseline: use the previous day's count as the prediction.
    baseline_predictions = test["LAG_1"].clip(lower=0)

    baseline_mae = mean_absolute_error(
        test[target], baseline_predictions
    )

    baseline_rmse = mean_squared_error(
        test[target], baseline_predictions
    ) ** 0.5

    metrics = {
        "model": "Random Forest Regressor",
        "split_date": str(split_date.date()),
        "training_rows": len(train),
        "testing_rows": len(test),
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "Baseline_MAE": round(float(baseline_mae), 4),
        "Baseline_RMSE": round(float(baseline_rmse), 4),
    }

    joblib.dump(model, MODEL_DIR / "historical_risk_model.joblib")
    pd.DataFrame([metrics]).to_csv(
        OUTPUT_DIR / "model_metrics.csv",
        index=False,
    )

    test_results = test[
        ["DATE", "AREA NAME", target]
    ].copy()

    test_results["PREDICTED_COUNT"] = predictions
    test_results.to_csv(
        OUTPUT_DIR / "historical_test_predictions.csv",
        index=False,
    )

    print("\nTRAINING COMPLETE")
    print("-----------------")
    print(f"Date range: {daily['DATE'].min().date()} to {daily['DATE'].max().date()}")
    print(f"Training rows: {len(train)}")
    print(f"Testing rows: {len(test)}")
    print(f"MAE: {mae:.3f} incidents")
    print(f"RMSE: {rmse:.3f} incidents")
    print(f"Model saved: {MODEL_DIR / 'historical_risk_model.joblib'}")
    print(f"Metrics saved: {OUTPUT_DIR / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()