from __future__ import annotations

from contextlib import asynccontextmanager

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

model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
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
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    # TODO: this should return ok only after the model is really loaded.
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
