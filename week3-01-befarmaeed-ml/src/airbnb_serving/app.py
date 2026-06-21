from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, Request

from .predictor import predict_batch as predict_batch_records
from .predictor import predict_single
from .schema import ListingFeatures, PredictionResponse

DEFAULT_TRACKING_URI = "http://185.50.38.163:33014"


def _load_model_from_env() -> tuple[Any, str]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    run_id = os.getenv("MODEL_RUN_ID")

    if not run_id:
        raise RuntimeError("MODEL_RUN_ID must be set before starting the service.")

    if tracking_username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = tracking_username
    if tracking_password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking_password

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri), run_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, run_id = _load_model_from_env()
    app.state.model = model
    app.state.model_run_id = run_id
    yield


app = FastAPI(
    title="Airbnb Demand Model Serving API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {
        "status": "ok",
        "model_run_id": request.app.state.model_run_id,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ListingFeatures, request: Request) -> PredictionResponse:
    return predict_single(
        features=payload,
        model=request.app.state.model,
        run_id=request.app.state.model_run_id,
    )


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(
    payload: list[ListingFeatures],
    request: Request,
) -> list[PredictionResponse]:
    return predict_batch_records(
        features_list=payload,
        model=request.app.state.model,
        run_id=request.app.state.model_run_id,
    )
