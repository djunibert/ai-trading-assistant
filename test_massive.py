import requests
import pandas as pd

API_KEY = "rpdBIbBber4S0rcAbdBH2MI1UQf7d9uE"
ticker = "GCJ5"

url = f"https://api.massive.com/futures/v1/aggs/{ticker}"

params = {
    "resolution": "5min",
    "window_start.gte": "2025-04-01",
    "window_start.lte": "2025-04-28",
    "limit": 50000,
    "sort": "window_start.asc",
    "apiKey": API_KEY
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

data = response.json()
df = pd.DataFrame(data.get("results", []))

df["datetime"] = (
    pd.to_datetime(df["window_start"], unit="ns", utc=True)
    .dt.tz_convert("America/Toronto")
)

print(df[
    [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]
].head())