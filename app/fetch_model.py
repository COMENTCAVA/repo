"""Télécharge la dernière version du modèle enregistré dans le registry MLflow (DagsHub).

Variables d'environnement lues par MLflow : MLFLOW_TRACKING_URI,
MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD (token DagsHub).
"""
import os

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = os.environ.get("MODEL_NAME", "RandomForest_Digits")
DEST = os.environ.get("MODEL_DIR", "/app/model")

client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    raise SystemExit(f"Aucune version trouvée pour le modèle {MODEL_NAME}")
latest = max(versions, key=lambda v: int(v.version))
print(f"Dernière version de {MODEL_NAME} : {latest.version}")

path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{MODEL_NAME}/{latest.version}", dst_path=DEST
)
print(f"Modèle téléchargé dans {path}")
