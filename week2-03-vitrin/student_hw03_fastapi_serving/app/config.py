from __future__ import annotations

import os
from pathlib import Path
from typing import List

APP_TITLE = "QBC12 HW03 Listing Availability Prediction API"
APP_VERSION = "0.2.0"

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = Path(__file__).resolve().parents[3]
SELECTED_RUN_ID = "d97de4e1b1334f1a8996b84d582616f6"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://185.50.38.163:33014")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
STUDENT_USERNAME = os.getenv(
    "STUDENT_USERNAME",
    MLFLOW_TRACKING_USERNAME or "student_amirhossein_sa",
)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", f"qbc12_hw02_{STUDENT_USERNAME}")
MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID", SELECTED_RUN_ID).strip()
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
LOCAL_MODEL_PATH = Path(
    os.getenv(
        "LOCAL_MODEL_PATH",
        REPOSITORY_DIR
        / "week2-02-az-model"
        / "outputs"
        / "mlflow_models"
        / SELECTED_RUN_ID,
    )
).expanduser()

DATASET_VERSION = "v1_student"
TARGET_NAME = "high_demand_proxy"
POSITIVE_LABEL = "high_demand_proxy"
NEGATIVE_LABEL = "not_high_demand_proxy"

# Public API feature order used to build the prediction DataFrame.
EXPECTED_FEATURE_COLUMNS: List[str] = [
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

# Audit, future-window, and target fields that must never be model inputs.
FORBIDDEN_FIELDS: List[str] = [
    "listing_id",
    "cutoff_date",
    "dataset_version",
    "future_calendar_days_observed_30d",
    "future_available_days_30d",
    "future_available_rate_30d",
    "high_demand_proxy",
]
