"""
Module : mlflow_manager.py

Description
-----------
Gestionnaire MLflow pour centraliser le suivi des expériences,
des paramètres, des métriques et des modèles.

Auteur : Junior Hébert
Projet : AI Trading System
"""

import mlflow
import mlflow.sklearn


class MLflowManager:
    """
    Classe utilitaire pour gérer MLflow.
    """

    def __init__(self, experiment_name: str):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str):
        """
        Démarre un run MLflow.
        """
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict) -> None:
        """
        Enregistre les paramètres dans MLflow.
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict) -> None:
        """
        Enregistre les métriques dans MLflow.
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    def log_sklearn_model(self, model, model_name: str) -> None:
        """
        Enregistre un modèle Scikit-Learn dans MLflow.
        """
        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
        )
