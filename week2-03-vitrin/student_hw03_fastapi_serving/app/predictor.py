from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable, List

import numpy as np
import pandas as pd
from fastapi import HTTPException, status

from . import config
from .schemas import ListingFeatures, PredictionResponse


def records_to_dataframe(records: Iterable[ListingFeatures]) -> pd.DataFrame:
    """Convert validated API payloads into the exact DataFrame expected by the model."""
    rows = [_record_to_dict(record) for record in records]

    for row in rows:
        provided_fields = set(row)
        forbidden_fields = sorted(provided_fields.intersection(config.FORBIDDEN_FIELDS))
        if forbidden_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Forbidden leakage or audit fields were provided.",
                    "forbidden_fields": forbidden_fields,
                },
            )

        unknown_fields = sorted(provided_fields.difference(config.EXPECTED_FEATURE_COLUMNS))
        if unknown_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Unknown model input fields were provided.",
                    "unknown_fields": unknown_fields,
                },
            )

    df = pd.DataFrame(rows)

    missing_cols = [c for c in config.EXPECTED_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Required model input fields are missing.",
                "missing_fields": missing_cols,
            },
        )

    return df[config.EXPECTED_FEATURE_COLUMNS]


def predict_records(model, records: List[ListingFeatures]) -> List[PredictionResponse]:
    """Run binary classification and return one response per input record."""
    X = records_to_dataframe(records)
    model_input = _prepare_model_input(model, X)

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(model_input))
        if probabilities.ndim != 2 or probabilities.shape[0] != len(model_input):
            raise RuntimeError("Model returned an invalid predict_proba shape.")

        positive_index = _positive_class_index(model, probabilities.shape[1])
        positive_probabilities = probabilities[:, positive_index].astype(float)
        predictions = (
            positive_probabilities >= config.PREDICTION_THRESHOLD
        ).astype(int)
        response_probabilities: list[float | None] = positive_probabilities.tolist()
    else:
        raw_predictions = np.asarray(model.predict(model_input)).reshape(-1)
        if len(raw_predictions) != len(model_input):
            raise RuntimeError("Model returned an invalid prediction shape.")
        predictions = (raw_predictions == 1).astype(int)
        response_probabilities = [None] * len(model_input)

    return [
        PredictionResponse(
            prediction=int(prediction),
            prediction_label=(
                config.POSITIVE_LABEL if prediction == 1 else config.NEGATIVE_LABEL
            ),
            probability=probability,
            threshold=config.PREDICTION_THRESHOLD,
        )
        for prediction, probability in zip(
            predictions, response_probabilities, strict=True
        )
    ]


def _record_to_dict(record: ListingFeatures | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, ListingFeatures):
        return record.model_dump()
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError("Each prediction record must be ListingFeatures or a mapping.")


def _prepare_model_input(model: Any, api_features: pd.DataFrame) -> pd.DataFrame:
    model_columns = getattr(model, "feature_names_in_", None)
    if model_columns is None:
        return api_features

    expected_model_columns = [str(column) for column in model_columns]
    model_input = api_features.copy()

    if "has_reviews_before_cutoff" in expected_model_columns:
        model_input["has_reviews_before_cutoff"] = (
            model_input["total_reviews_before_cutoff"].fillna(0).gt(0).astype(int)
        )

    unsupported_columns = [
        column
        for column in expected_model_columns
        if column not in model_input.columns
    ]
    if unsupported_columns:
        raise RuntimeError(
            "The loaded model expects unsupported features: " + ", ".join(unsupported_columns)
        )

    return model_input[expected_model_columns]


def _positive_class_index(model: Any, probability_column_count: int) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None:
        if probability_column_count == 2:
            return 1
        raise RuntimeError("Cannot identify the positive class probability column.")

    class_values = np.asarray(classes)
    positive_matches = np.flatnonzero(class_values == 1)
    if len(positive_matches) != 1:
        raise RuntimeError("Loaded model does not expose exactly one positive class 1.")
    return int(positive_matches[0])
