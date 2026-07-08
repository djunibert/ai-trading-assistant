from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.utils.logger import get_logger


logger = get_logger(__name__)

DATASET_PATH = Path("data/final/dataset_ml_v2.csv")
MODEL_PATH = Path("models/xgboost_v2.pkl")
REPORTS_DIR = Path("reports/evaluation")


def evaluate_xgboost_v2() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    features = bundle["features"]
    inverse_mapping = bundle["inverse_target_mapping"]

    X = df[features]
    y_true = df["target"]

    y_pred_encoded = model.predict(X)
    y_pred = pd.Series(y_pred_encoded).map(inverse_mapping)

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    logger.info(f"Accuracy : {accuracy:.4f}")
    logger.info(f"F1 macro : {f1_macro:.4f}")
    logger.info(f"F1 weighted : {f1_weighted:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORTS_DIR / "classification_report_xgboost_v2.csv",
        encoding="utf-8-sig",
    )

    labels = [-1, 0, 1]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ).to_csv(
        REPORTS_DIR / "confusion_matrix_xgboost_v2.csv",
        encoding="utf-8-sig",
    )

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        REPORTS_DIR / "feature_importance_xgboost_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("Évaluation XGBoost V2 terminée.")


if __name__ == "__main__":
    evaluate_xgboost_v2()