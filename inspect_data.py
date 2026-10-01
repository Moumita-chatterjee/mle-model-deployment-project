import pandas as pd

file_path = "data/yellow_taxi_tripdata_2025-01.parquet"

df = pd.read_parquet(file_path)
#print(df.shape)
#print(df.columns)
#print(df.head())

#print(df.info())
#print(df.isnull().sum())

df["duration"] = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds()/60

before_cleaning = len(df)
df = df[(df["duration"] > 0) & (df["duration"] <= 60)]
after_cleaning = len(df)

print(f"Rows before cleaning: {before_cleaning:,}")
print(f"Rows after cleaning:  {after_cleaning:,}")
print(f"Rows removed:         {before_cleaning - after_cleaning:,}")


print(df["duration"].describe())

before_distance = len(df)

df = df[df["trip_distance"] <= 100]

after_distance = len(df)

print(f"Rows before distance cleaning: {before_distance:,}")
print(f"Rows after distance cleaning:  {after_distance:,}")
print(f"Rows removed:                  {before_distance - after_distance:,}")
# print("Negative duration:", (df["duration"] < 0).sum())
# print("zero duration:", (df["duration"] == 0).sum())
# print("long duration:", (df["duration"] > 120).sum())

# print("\nNegative duration examples:")
# print(
#     df.loc[
#         df["duration"] <= 0,
#         [
#             "tpep_pickup_datetime",
#             "tpep_dropoff_datetime",
#             "duration",
#             "trip_distance",
#         ],
#     ].head(10)
# )

# print("\nLong duration examples:")
# print(
#     df.loc[
#         df["duration"] > 120,
#         [
#             "tpep_pickup_datetime",
#             "tpep_dropoff_datetime",
#             "duration",
#             "trip_distance",
#         ],
#     ].head(10)
# )

# print(
#     df[
#         ["duration", "trip_distance"]
#     ].sort_values("duration", ascending=False).head(20)
# )

# df = df[df["duration"] > 0]
# print("99th percentile:", df["duration"].quantile(0.99))
# print("95th percentile:", df["duration"].quantile(0.95))

#feature checking
# print(df[
#     ["trip_distance","passenger_count","PULocationID","DOLocationID"]
# ].describe())

# print("Zero distance:", (df["trip_distance"] == 0).sum())
# print("Distance > 100 miles:", (df["trip_distance"] > 100).sum())
# print("Distance > 1000 miles:", (df["trip_distance"] > 1000).sum())

# print(
#     df[["trip_distance", "duration"]]
#     .sort_values("trip_distance", ascending=False)
#     .head(20)
# )

# print("Zero distance + duration > 0:",
#       ((df["trip_distance"] == 0) & (df["duration"] > 0)).sum())

# print("Zero distance + duration <= 1 min:",
#       ((df["trip_distance"] == 0) & (df["duration"] <= 1)).sum())

# print(
#     df[df["trip_distance"] == 0][
#         ["trip_distance", "duration", "passenger_count",
#          "PULocationID", "DOLocationID"]
#     ]
#     .describe()
# )

# print("Distance > 50 miles:", (df["trip_distance"] > 50).sum())
# print("Distance > 100 miles:", (df["trip_distance"] > 100).sum())

# print(
#     df[df["trip_distance"] > 50][
#         ["trip_distance", "duration"]
#     ]
#     .sort_values("trip_distance", ascending=False)
#     .tail(20)
# )

print("Missing passenger_count:",
      df["passenger_count"].isna().sum())

print("Zero passenger_count:",
      (df["passenger_count"] == 0).sum())

print("Passenger count distribution:")
print(df["passenger_count"].value_counts(dropna=False).sort_index())

df["passenger_count"] = df["passenger_count"].fillna(
    df["passenger_count"].median()
)



df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour

print(
    df[
        [
            "trip_distance",
            "passenger_count",
            "PULocationID",
            "DOLocationID",
            "pickup_hour",
            "duration",
        ]
    ].head()
)

print("Missing passenger_count after filling:",
      df["passenger_count"].isna().sum())

print(f"Final dataset shape: {df.shape}")