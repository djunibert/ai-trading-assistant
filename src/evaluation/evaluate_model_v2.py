"""
Évaluation du modèle Random Forest V2.

Génère :
- classification_report.csv
- confusion_matrix.csv
- feature_importance.csv
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.utils.logger import get_logger


logger = get_logger(__name__)

DATASET_PATH = Path("data/final/dataset_ml_v2.csv")
MODEL_PATH = Path("models/random_forest_v2.pkl")
REPORTS_DIR = Path("reports/evaluation")


def evaluate_model_v2() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Chargement du dataset ML V2")
    df = pd.read_csv(DATASET_PATH)

    logger.info("Chargement du modèle Random Forest V2")
    model_bundle = joblib.load(MODEL_PATH)

    model = model_bundle["model"]
    features = model_bundle["features"]

    X = df[features]
    y = df["target"]

    logger.info("Prédiction sur le dataset complet")
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")

    logger.info(f"Accuracy : {accuracy:.4f}")
    logger.info(f"F1 macro : {f1_macro:.4f}")
    logger.info(f"F1 weighted : {f1_weighted:.4f}")

    report_dict = classification_report(
        y,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()

    report_df.to_csv(
        REPORTS_DIR / "classification_report_v2.csv",
        encoding="utf-8-sig",
    )

    labels = [-1, 0, 1]

    cm = confusion_matrix(
        y,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    cm_df.to_csv(
        REPORTS_DIR / "confusion_matrix_v2.csv",
        encoding="utf-8-sig",
    )

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_,
        }).sort_values(
            by="importance",
            ascending=False,
        )

        importance_df.to_csv(
            REPORTS_DIR / "feature_importance_v2.csv",
            index=False,
            encoding="utf-8-sig",
        )

        logger.info("Feature importance sauvegardée.")

    logger.info(f"Rapports sauvegardés dans : {REPORTS_DIR}")
    logger.info("Évaluation terminée avec succès.")


if __name__ == "__main__":
    evaluate_model_v2()