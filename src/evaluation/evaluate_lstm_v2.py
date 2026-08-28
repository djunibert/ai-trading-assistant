"""
Évaluation du modèle LSTM V2.
"""

from pathlib import Path

#import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.deep_learning.sequence_builder import SequenceBuilder
from src.utils.logger import get_logger


logger = get_logger(__name__)

DATASET_PATH = Path("data/final/dataset_lstm_v2.csv")
MODEL_PATH = Path("models/lstm_v2.keras")
REPORTS_DIR = Path("reports/evaluation")

SEQUENCE_LENGTH = 32


def evaluate_lstm_v2() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)

    builder = SequenceBuilder(sequence_length=SEQUENCE_LENGTH)

    X, y, features, _ = builder.build(
        df,
        target_column="target",
        columns_to_exclude=["future_return_1"],
    )

    target_mapping = {-1: 0, 0: 1, 1: 2}
    inverse_mapping = {0: -1, 1: 0, 2: 1}

    y_encoded = np.vectorize(target_mapping.get)(y)

    model = load_model(MODEL_PATH)

    y_proba = model.predict(X)
    y_pred_encoded = np.argmax(y_proba, axis=1)

    y_true = pd.Series(y_encoded).map(inverse_mapping)
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
        REPORTS_DIR / "classification_report_lstm_v2.csv",
        encoding="utf-8-sig",
    )

    labels = [-1, 0, 1]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ).to_csv(
        REPORTS_DIR / "confusion_matrix_lstm_v2.csv",
        encoding="utf-8-sig",
    )

    logger.info("Évaluation LSTM V2 terminée.")


if __name__ == "__main__":
    evaluate_lstm_v2()