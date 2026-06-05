import pandas as pd

REQUIRED_LISTING_COLUMNS = {
    "listing_id",
    "neighbourhood",
    "price",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
}

REQUIRED_SEGMENT_COLUMNS = {"neighbourhood", "tourism_segment", "priority_level"}

OUTPUT_COLUMNS = [
    "neighbourhood",
    "num_listings",
    "avg_price",
    "median_price",
    "avg_minimum_nights",
    "availability_365_avg",
    "total_reviews",
    "reviews_per_listing",
    "tourism_segment",
    "priority_level",
]


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def build_neighbourhood_summary(listings: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    _require_columns(listings, REQUIRED_LISTING_COLUMNS, "listings")
    _require_columns(segments, REQUIRED_SEGMENT_COLUMNS, "segments")

    summary = (
        listings.groupby("neighbourhood", as_index=False)
        .agg(
            num_listings=("listing_id", "count"),
            avg_price=("price", "mean"),
            median_price=("price", "median"),
            avg_minimum_nights=("minimum_nights", "mean"),
            availability_365_avg=("availability_365", "mean"),
            total_reviews=("number_of_reviews", "sum"),
        )
    )

    summary["reviews_per_listing"] = summary["total_reviews"] / summary["num_listings"]

    summary = summary.merge(
        segments[["neighbourhood", "tourism_segment", "priority_level"]],
        on="neighbourhood",
        how="left",
    )

    summary[["tourism_segment", "priority_level"]] = summary[
        ["tourism_segment", "priority_level"]
    ].fillna("unknown")

    numeric_columns = [
        "avg_price",
        "median_price",
        "avg_minimum_nights",
        "availability_365_avg",
        "reviews_per_listing",
    ]
    summary[numeric_columns] = summary[numeric_columns].round(2)

    return summary[OUTPUT_COLUMNS].sort_values("neighbourhood").reset_index(drop=True)
