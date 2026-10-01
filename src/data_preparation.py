from pathlib import Path
import pandas as pd

def load_data(file_path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(file_path)

def create_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration"] = (
        df["tpep_dropoff_datetime"]
        - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid duration and extreme distance values."""
    df = df.copy()

    df = df[
        (df["duration"] > 0)
        & (df["duration"] <= 60)
    ]

    df = df[df["trip_distance"] <= 100]

    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features used by the baseline model."""
    df = df.copy()

    df["passenger_count"] = df["passenger_count"].fillna(
        df["passenger_count"].median()
    )

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

    return df
