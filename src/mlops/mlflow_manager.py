"""
Gestionnaire MLflow.

Ce fichier centralise toute la logique MLflow :
- nom de l'expérience
- démarrage du run
- paramètres
- métriques
- sauvegarde du modèle dans MLflow
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost

class MLflowManager:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict) -> None:
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict) -> None:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    def log_sklearn_model(self, model, artifact_name: str) -> None:
        mlflow.sklearn.log_model(
            sk_model=model,
            name=artifact_name,
        )
        
        
    def log_xgboost_model(self, model, artifact_name: str) -> None:
        mlflow.xgboost.log_model(
            xgb_model=model,
            name=artifact_name,
        )