import pandas as pd
from airbnb_ops.transform import OUTPUT_COLUMNS

PII_COLUMNS = {"host_name", "host_id", "reviewer_name", "reviewer_id", "listing_url", "host_url"}


def validate_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        raise ValueError("Output summary is empty")

    missing = set(OUTPUT_COLUMNS) - set(summary.columns)
    if missing:
        raise ValueError(f"Output is missing required columns: {sorted(missing)}")

    leaked = PII_COLUMNS & set(summary.columns)
    if leaked:
        raise ValueError(f"PII leaked into output: {sorted(leaked)}")

    if summary["neighbourhood"].isna().any():
        raise ValueError("neighbourhood contains null values")

    if not (summary["num_listings"] > 0).all():
        raise ValueError("num_listings must be greater than 0")

    if not (summary["avg_price"] >= 0).all():
        raise ValueError("avg_price must be non-negative")

    if not summary["availability_365_avg"].between(0, 365).all():
        raise ValueError("availability_365_avg must be between 0 and 365")
