"""
Comparaison des modèles V2.

Compare :
- Random Forest V2
- XGBoost V2
- LSTM V2
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)

REPORTS_DIR = Path("reports/evaluation")
OUTPUT_FILE = REPORTS_DIR / "model_comparison_v2.csv"


MODELS = {
    "Random Forest V2": "classification_report_v2.csv",
    "XGBoost V2": "classification_report_xgboost_v2.csv",
    "LSTM V2": "classification_report_lstm_v2.csv",
}


def extract_metrics(model_name: str, report_path: Path) -> dict | None:
    if not report_path.exists():
        logger.warning(f"Rapport manquant : {report_path}")
        return None

    report_df = pd.read_csv(report_path, index_col=0)

    return {
        "model": model_name,
        "accuracy": round(float(report_df.loc["accuracy", "f1-score"]), 4),
        "f1_macro": round(float(report_df.loc["macro avg", "f1-score"]), 4),
        "f1_weighted": round(float(report_df.loc["weighted avg", "f1-score"]), 4),
        "precision_macro": round(float(report_df.loc["macro avg", "precision"]), 4),
        "recall_macro": round(float(report_df.loc["macro avg", "recall"]), 4),
    }


def compare_models_v2() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for model_name, file_name in MODELS.items():
        metrics = extract_metrics(
            model_name=model_name,
            report_path=REPORTS_DIR / file_name,
        )

        if metrics is not None:
            results.append(metrics)

    if not results:
        raise ValueError("Aucun rapport trouvé pour comparer les modèles.")

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df.sort_values(
        by=["f1_macro", "accuracy"],
        ascending=False,
    )

    comparison_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("Comparaison des modèles V2")
    logger.info(f"\n{comparison_df}")
    logger.info(f"Rapport sauvegardé : {OUTPUT_FILE}")


if __name__ == "__main__":
    compare_models_v2()