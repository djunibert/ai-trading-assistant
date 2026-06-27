import os
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("THENEWSAPI_KEY")

QUERY = (
    'gold OR silver OR inflation OR CPI OR '
    '"Federal Reserve" OR Fed OR FOMC OR '
    '"Nonfarm Payrolls" OR NFP OR '
    '"interest rates" OR "US dollar" OR DXY OR '
    'geopolitical OR war OR conflict'
)

START_DATE = "2024-01-01"
END_DATE = "2026-06-06"


def collect_thenewsapi_data():
    if not API_KEY:
        raise ValueError("THENEWSAPI_KEY manquant dans le fichier .env")

    url = "https://api.thenewsapi.com/v1/news/all"

    params = {
        "api_token": API_KEY,
        "search": QUERY,
        "language": "en",
        "published_after": START_DATE,
        "published_before": END_DATE,
        "limit": 100,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    json_data = response.json()
    articles = json_data.get("data", [])

    rows = []

    for article in articles:
        rows.append({
            "datetime": article.get("published_at"),
            "source": article.get("source"),
            "title": article.get("title"),
            "description": article.get("description"),
            "url": article.get("url"),
            "language": article.get("language"),
            "categories": ",".join(article.get("categories", [])) if article.get("categories") else None,
        })

    df = pd.DataFrame(rows)

    output_dir = Path("data/raw/news")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "thenewsapi_articles.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Fichier créé : {output_path}")
    print(df.head())


if __name__ == "__main__":
    collect_thenewsapi_data()