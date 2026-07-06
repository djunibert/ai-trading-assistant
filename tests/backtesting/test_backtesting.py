import pandas as pd

from src.pipelines.market_structure_pipeline import MarketStructurePipeline
from src.backtesting.backtesting_engine import BacktestingEngine
from src.utils.logger import get_logger


logger = get_logger(__name__)

DATA_PATH = "data/features/market_macro/1h/gold.csv"
TRADES_OUTPUT = "reports/backtesting_trades.csv"
STATS_OUTPUT = "reports/backtesting_statistics.csv"


def main() -> None:
    logger.info("=" * 80)
    logger.info("BACKTESTING V2")
    logger.info("=" * 80)

    df = pd.read_csv(DATA_PATH)

    pipeline = MarketStructurePipeline()
    df = pipeline.run(df)

    logger.info("Pipeline Market Structure exécuté.")

    result = BacktestingEngine().run(df)

    statistics = result["statistics"]
    trades_df = result["trades"]

    logger.info(f"Total trades : {statistics['total_trades']}")
    logger.info(f"Wins : {statistics['wins']}")
    logger.info(f"Losses : {statistics['losses']}")
    logger.info(f"Win rate : {statistics['win_rate']} %")
    logger.info(f"Total PnL : {statistics['total_pnl']}")
    logger.info(f"Profit factor : {statistics['profit_factor']}")
    logger.info(f"Expectancy : {statistics['expectancy']}")
    logger.info(f"Max drawdown : {statistics['max_drawdown']}")

    if not trades_df.empty:
        trades_df.to_csv(TRADES_OUTPUT, index=False, encoding="utf-8-sig")
        pd.DataFrame([statistics]).to_csv(
            STATS_OUTPUT,
            index=False,
            encoding="utf-8-sig"
        )

        logger.info(f"Trades sauvegardés : {TRADES_OUTPUT}")
        logger.info(f"Statistiques sauvegardées : {STATS_OUTPUT}")

        print(trades_df.tail(20))
    else:
        logger.warning("Aucun trade généré.")

    logger.info("Backtesting V2 terminé avec succès.")


if __name__ == "__main__":
    main()