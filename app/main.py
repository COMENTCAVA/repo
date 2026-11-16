import os

import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")

# Une seule route publique : /predict (pas de /docs, /redoc ni /openapi.json)
app = FastAPI(title="Digits classifier", docs_url=None, redoc_url=None, openapi_url=None)

model = mlflow.sklearn.load_model(MODEL_DIR)


class PredictRequest(BaseModel):
    # Une image 8x8 aplatie : 64 valeurs (jeu de données sklearn load_digits)
    features: list[float] = Field(..., min_length=64, max_length=64)


@app.post("/predict")
def predict(request: PredictRequest):
    prediction = model.predict([request.features])[0]
    return {"prediction": int(prediction)}
