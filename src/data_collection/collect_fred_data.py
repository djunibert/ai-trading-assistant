import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")

SERIES = {
    "cpi": "CPIAUCSL",
    "nfp": "PAYEMS",
    "fed_rate": "FEDFUNDS",
    "unemployment_rate": "UNRATE",
}


def collect_fred_series(name: str, series_id: str) -> pd.DataFrame:
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY est manquant. Vérifie ton fichier .env")

    url = "https://api.stlouisfed.org/fred/series/observations"

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "2020-01-01",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()["observations"]

    df = pd.DataFrame(data)
    df = df[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.rename(columns={
        "date": "datetime",
        "value": name
    })

    return df


def collect_fred_data() -> None:
    output_dir = Path("data/raw/macro")
    output_dir.mkdir(parents=True, exist_ok=True)

    final_df = None

    for name, series_id in SERIES.items():
        print(f"Collecte de {name} ({series_id})...")
        df = collect_fred_series(name, series_id)

        if final_df is None:
            final_df = df
        else:
            final_df = final_df.merge(df, on="datetime", how="outer")

    final_df = final_df.sort_values("datetime")

    output_path = output_dir / "macro_fred.csv"
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Fichier créé : {output_path}")


if __name__ == "__main__":
    collect_fred_data()