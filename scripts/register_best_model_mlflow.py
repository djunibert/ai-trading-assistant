"""
Enregistre rapidement le meilleur modèle dans MLflow.

Modèle choisi :
GC 5m Random Forest V3
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn


MODEL_PATH = Path(
    "models/GC/5m/random_forest/random_forest_v3.pkl"
)

EXPERIMENT_NAME = "AI Trading System V3"
REGISTERED_MODEL_NAME = "GC_RandomForest_5m"


def register_model() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    package = joblib.load(MODEL_PATH)

    model = package["model"]
    features = package["features"]
    metrics = package.get("metrics", {})

    mlflow.set_tracking_uri(
        "http://localhost:5000"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="random_forest_v3_GC_5m"
    ) as run:
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "symbol": "GC",
            "timeframe": "5m",
            "feature_count": len(features),
            "split_type": package.get(
                "split_type",
                "chronological",
            ),
            "train_ratio": package.get(
                "train_ratio",
                0.70,
            ),
            "validation_ratio": package.get(
                "validation_ratio",
                0.15,
            ),
            "test_ratio": package.get(
                "test_ratio",
                0.15,
            ),
        })

        for metric_name, metric_value in metrics.items():
            if isinstance(
                metric_value,
                (int, float),
            ):
                mlflow.log_metric(
                    metric_name,
                    float(metric_value),
                )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=None,
        )

        mlflow.log_dict(
            {
                "features": features,
            },
            "features.json",
        )

        print(
            f"Run créé : {run.info.run_id}"
        )

        print(
            f"Modèle enregistré : "
            f"{REGISTERED_MODEL_NAME}"
        )


if __name__ == "__main__":
    register_model()