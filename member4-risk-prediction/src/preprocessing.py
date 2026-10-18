import pandas as pd


def load_and_filter_shoplifting(file_path):
    """
    Load crime data and keep only shoplifting incidents.
    """

    useful_columns = [
        "DATE OCC",
        "TIME OCC",
        "AREA NAME",
        "Rpt Dist No",
        "Crm Cd Desc",
        "Premis Desc",
        "LAT",
        "LON"
    ]

    df = pd.read_csv(file_path, usecols=useful_columns)

    shoplifting_df = df[
        df["Crm Cd Desc"].str.contains(
            "SHOPLIFTING",
            case=False,
            na=False
        )
    ].copy()

    return shoplifting_df


def create_time_features(df):
    """
    Create useful time-based features from incident date and time.
    """

    df["DATE OCC"] = pd.to_datetime(
        df["DATE OCC"],
        errors="coerce"
    )

    df["YEAR"] = df["DATE OCC"].dt.year
    df["MONTH"] = df["DATE OCC"].dt.month
    df["DAY_OF_WEEK"] = df["DATE OCC"].dt.day_name()

    df["HOUR"] = df["TIME OCC"] // 100

    return df