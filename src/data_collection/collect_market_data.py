from pathlib import Path
import yfinance as yf


MARKET_SYMBOLS = {
    "gold": "GC=F",
    "silver": "SI=F",
    "nasdaq": "NQ=F",
    "sp500": "ES=F",
}


def collect_market_data(start_date="2020-01-01", interval="1d"):
    output_dir = Path("data/raw/market")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, symbol in MARKET_SYMBOLS.items():
        print(f"Collecte de {name} ({symbol})...")

        df = yf.download(
            symbol,
            start=start_date,
            interval=interval,
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            print(f"Aucune donnée trouvée pour {name}")
            continue

        df = df.reset_index()
        output_path = output_dir / f"{name}.csv"
        df.to_csv(output_path, index=False)

        print(f"Fichier créé : {output_path}")


if __name__ == "__main__":
    collect_market_data()