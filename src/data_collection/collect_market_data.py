import yfinance as yf

from src.utils.config import MARKET_SYMBOLS, START_DATE
from src.utils.paths import RAW_MARKET_DIR
from src.utils.logger import get_logger


logger = get_logger(__name__)


def collect_market_data(interval: str = "1d") -> None:
    RAW_MARKET_DIR.mkdir(parents=True, exist_ok=True)

    for name, symbol in MARKET_SYMBOLS.items():
        logger.info(f"Collecte de {name} ({symbol})")

        df = yf.download(
            symbol,
            start=START_DATE,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            logger.warning(f"Aucune donnée trouvée pour {name}")
            continue

        df = df.reset_index()

        output_path = RAW_MARKET_DIR / f"{name}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        logger.info(f"Fichier créé : {output_path}")


if __name__ == "__main__":
    collect_market_data()