from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .schema import ListingFeatures, PredictionResponse

FEATURE_COLUMNS = [
    "room_type",
    "property_type",
    "neighbourhood_name",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "listing_price",
    "minimum_nights",
    "maximum_nights",
    "instant_bookable",
    "host_is_superhost",
    "host_listing_count",
    "total_reviews_before_cutoff",
    "unique_reviewers_before_cutoff",
    "avg_comment_len_before_cutoff",
    "max_comment_len_before_cutoff",
    "days_since_last_review",
    "available_days_last_90d",
    "available_rate_last_90d",
    "avg_minimum_nights_calendar_last_90d",
    "avg_maximum_nights_calendar_last_90d",
    "available_days_last_30d",
    "available_rate_last_30d",
    "avg_minimum_nights_calendar_last_30d",
    "avg_maximum_nights_calendar_last_30d",
]


def predict_single(
    features: ListingFeatures,
    model: Any,
    run_id: str,
) -> PredictionResponse:
    frame, listing_ids = _features_to_frame([features])
    predictions, probabilities = _predict_frame(frame, model)
    return _build_response(
        listing_id=listing_ids[0],
        prediction=predictions[0],
        probability=probabilities[0],
        run_id=run_id,
    )


def predict_batch(
    features_list: list[ListingFeatures],
    model: Any,
    run_id: str,
) -> list[PredictionResponse]:
    if not features_list:
        return []

    frame, listing_ids = _features_to_frame(features_list)
    predictions, probabilities = _predict_frame(frame, model)
    return [
        _build_response(
            listing_id=listing_id,
            prediction=prediction,
            probability=probability,
            run_id=run_id,
        )
        for listing_id, prediction, probability in zip(
            listing_ids,
            predictions,
            probabilities,
            strict=True,
        )
    ]


def _features_to_frame(
    features_list: list[ListingFeatures],
) -> tuple[pd.DataFrame, list[int | None]]:
    rows: list[dict[str, Any]] = []
    listing_ids: list[int | None] = []

    for features in features_list:
        row = features.model_dump()
        listing_ids.append(row.pop("listing_id", None))
        rows.append(row)

    frame = pd.DataFrame(rows)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError("Missing model input columns: " + ", ".join(missing_columns))

    return frame[FEATURE_COLUMNS], listing_ids


def _predict_frame(frame: pd.DataFrame, model: Any) -> tuple[list[int], list[float]]:
    model_input = _prepare_model_input(frame, model)
    predictions = np.asarray(model.predict(model_input)).reshape(-1)
    probabilities = np.asarray(model.predict_proba(model_input))

    if len(predictions) != len(model_input):
        raise RuntimeError("Model returned an invalid prediction shape.")
    if probabilities.ndim != 2 or probabilities.shape[0] != len(model_input):
        raise RuntimeError("Model returned an invalid probability shape.")

    positive_index = _positive_class_index(model, probabilities.shape[1])
    high_demand_probabilities = probabilities[:, positive_index]

    return (
        [int(value) for value in predictions.tolist()],
        [float(value) for value in high_demand_probabilities.tolist()],
    )


def _prepare_model_input(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
    model_input = _normalize_missing_values(frame)
    expected_columns = _expected_model_columns(model)

    if expected_columns is None:
        return model_input

    if "has_reviews_before_cutoff" in expected_columns:
        model_input["has_reviews_before_cutoff"] = (
            pd.to_numeric(
                model_input["total_reviews_before_cutoff"],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .astype(int)
        )

    missing_columns = [
        column for column in expected_columns if column not in model_input.columns
    ]
    if missing_columns:
        raise ValueError(
            "Loaded model expects columns that are not available from the API: "
            + ", ".join(missing_columns)
        )

    return model_input[expected_columns]


def _normalize_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(object).where(pd.notna(frame), np.nan)


def _expected_model_columns(model: Any) -> list[str] | None:
    columns = getattr(model, "feature_names_in_", None)
    if columns is None:
        return None
    return [str(column) for column in columns]


def _positive_class_index(model: Any, probability_column_count: int) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None:
        if probability_column_count == 2:
            return 1
        raise RuntimeError("Cannot infer the positive class probability column.")

    class_values = np.asarray(classes)
    positive_matches = np.flatnonzero(class_values == 1)
    if len(positive_matches) != 1:
        raise RuntimeError("Loaded model does not expose a single positive class 1.")
    return int(positive_matches[0])


def _build_response(
    listing_id: int | None,
    prediction: int,
    probability: float,
    run_id: str,
) -> PredictionResponse:
    return PredictionResponse(
        listing_id=listing_id,
        prediction=int(prediction),
        probability_high_demand=float(probability),
        model_run_id=str(run_id),
    )
