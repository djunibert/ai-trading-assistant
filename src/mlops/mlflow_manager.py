"""
Gestionnaire MLflow du projet AI Trading System.
"""

import mlflow
import mlflow.sklearn


class MLflowManager:
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict) -> None:
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict) -> None:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    def log_sklearn_model(self, model, model_name: str) -> None:
        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name
        )