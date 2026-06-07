from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, status

from . import config
from .model_loader import ModelService
from .predictor import predict_records
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ListingFeatures,
    ModelInfoResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)
model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting HW03 API %s from %s", config.APP_VERSION, Path(__file__).resolve())
    model_service.load()
    if model_service.state.loaded:
        logger.info(
            "Loaded run %s from %s",
            model_service.state.run_id,
            model_service.state.tags.get("model_source", "unknown"),
        )
    else:
        logger.error("Model loading failed: %s", model_service.state.error)
    yield


app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description="HW03 FastAPI service. Use Swagger at /docs.",
    lifespan=lifespan,
)


@app.get("/", tags=["service"])
def root() -> dict:
    return {
        "message": "QBC12 Listing Availability Prediction API",
        "version": config.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    if model_service.state.loaded:
        return HealthResponse(status="ok", model_loaded=True)
    return HealthResponse(status="error", model_loaded=False, error=model_service.state.error)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**model_service.model_info())


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: ListingFeatures) -> PredictionResponse:
    try:
        model = model_service.require_model()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded: {exc}",
        ) from exc

    return predict_records(model, [payload])[0]


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        model = model_service.require_model()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded: {exc}",
        ) from exc

    predictions = predict_records(model, payload.records)
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)
