import os
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

QUERY = (
    '(gold OR silver OR "Federal Reserve" OR Fed OR FOMC OR '
    'inflation OR CPI OR "Nonfarm Payrolls" OR NFP OR '
    '"interest rates" OR "US dollar" OR DXY OR '
    'war OR conflict OR geopolitical)'
)

FROM_DATE = "2025-06-01"
TO_DATE = "2026-06-21"


def collect_newsapi_data():
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY manquant dans le fichier .env")

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": QUERY,
        "searchIn": "title,description",
        "from": FROM_DATE,
        "to": TO_DATE,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": NEWS_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    articles = response.json().get("articles", [])

    rows = []
    for article in articles:
        rows.append({
            "datetime": article.get("publishedAt"),
            "source": article.get("source", {}).get("name"),
            "author": article.get("author"),
            "title": article.get("title"),
            "description": article.get("description"),
            "url": article.get("url"),
        })

    df = pd.DataFrame(rows)

    output_dir = Path("data/raw/news")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "newsapi_articles.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Fichier créé : {output_path}")
    print(df.head())


if __name__ == "__main__":
    collect_newsapi_data()
