from pathlib import Path
import requests

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
output_path = Path("data/yellow_taxi_tripdata_2025-01.parquet")
response = requests.get(URL)
response.raise_for_status()
output_path.write_bytes(response.content)

print(f"downloaded: {output_path}")