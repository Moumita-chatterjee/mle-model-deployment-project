from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import mlflow

from data_preparation import (
    load_data,
    create_duration,
    clean_data,
    create_features,
)


DATA_PATH = Path("data/yellow_taxi_tripdata_2025-01.parquet")

FEATURES = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
]

TARGET = "duration"


df = load_data(DATA_PATH)
df = create_duration(df)
df = clean_data(df)
df = create_features(df)

print("Prepared data shape:", df.shape)

X = df[FEATURES]
y = df[TARGET]

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)


mlflow.set_experiment("taxi-duration-baseline")
with mlflow.start_run():
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators",100)
    mlflow.log_param("random_state",42)
    mlflow.log_param("features",FEATURES)


    model.fit(X_train, y_train)
    print("Model training completed.")

    y_pred = model.predict(X_val)
    print("Predictions completed")

    rmse = root_mean_squared_error(y_val, y_pred)

    mlflow.log_metric("rmse", float(rmse))

    print(f"Validation RMSE: {rmse:.2f} minutes")

    mlflow.sklearn.log_model(model, "model")