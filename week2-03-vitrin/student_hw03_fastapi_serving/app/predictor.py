from __future__ import annotations

from typing import Iterable, List

import pandas as pd
from fastapi import HTTPException, status

from . import config
from .schemas import ListingFeatures, PredictionResponse


def records_to_dataframe(records: Iterable[ListingFeatures]) -> pd.DataFrame:
    """Convert validated API payloads into the exact DataFrame expected by the model."""
    rows = [record.model_dump() for record in records]
    df = pd.DataFrame(rows)

    # TODO 1: reject unknown fields and forbidden leakage fields.
    # TODO 2: check missing fields against config.EXPECTED_FEATURE_COLUMNS.
    # TODO 3: return df[config.EXPECTED_FEATURE_COLUMNS].

    missing_cols = [c for c in config.EXPECTED_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "TODO: missing feature handling", "missing_fields": missing_cols},
        )

    return df[config.EXPECTED_FEATURE_COLUMNS]


def predict_records(model, records: List[ListingFeatures]) -> List[PredictionResponse]:
    """TODO: Run model prediction and return API responses."""
    X = records_to_dataframe(records)

    # TODO:
    # - if model has predict_proba, use positive-class probability.
    # - apply config.PREDICTION_THRESHOLD.
    # - return one PredictionResponse per record.
    # Temporary placeholder so the endpoint shape is clear:
    return [
        PredictionResponse(
            prediction=0,
            prediction_label=config.NEGATIVE_LABEL,
            probability=None,
            threshold=config.PREDICTION_THRESHOLD,
        )
        for _ in range(len(X))
    ]
