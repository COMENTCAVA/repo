FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Au build : récupère le dernier modèle entraîné depuis DagsHub via MLflow.
# Le token est fourni comme "secret" BuildKit : il n'est écrit dans aucune couche de l'image.
ENV MLFLOW_TRACKING_URI=https://dagshub.com/cone/repo.mlflow
ENV MODEL_DIR=/app/model
RUN --mount=type=secret,id=dagshub_token \
    MLFLOW_TRACKING_USERNAME=cone \
    MLFLOW_TRACKING_PASSWORD="$(cat /run/secrets/dagshub_token)" \
    python app/fetch_model.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
