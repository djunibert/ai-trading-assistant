import mlflow
from mlflow import MlflowClient


mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()

models = client.search_registered_models()

if not models:
    print("Aucun modèle enregistré.")
else:
    for model in models:
        print(f"Modèle : {model.name}")

        versions = client.search_model_versions(
            f"name='{model.name}'"
        )

        for version in versions:
            print(
                f"Version : {version.version} | "
                f"Statut : {version.status} | "
                f"Run ID : {version.run_id}"
            )