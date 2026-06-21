from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ListingFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "listing_id": 12345,
                "room_type": "Entire home/apt",
                "property_type": "Entire rental unit",
                "neighbourhood_name": "Centrum-West",
                "accommodates": 2,
                "bedrooms": 1.0,
                "beds": 1.0,
                "bathrooms": 1.0,
                "listing_price": 150.0,
                "minimum_nights": 2,
                "maximum_nights": 365,
                "instant_bookable": True,
                "host_is_superhost": False,
                "host_listing_count": 1,
                "total_reviews_before_cutoff": 10.0,
                "unique_reviewers_before_cutoff": 9.0,
                "avg_comment_len_before_cutoff": 120.0,
                "max_comment_len_before_cutoff": 300.0,
                "days_since_last_review": 30.0,
                "available_days_last_90d": 45,
                "available_rate_last_90d": 0.5,
                "avg_minimum_nights_calendar_last_90d": 2.0,
                "avg_maximum_nights_calendar_last_90d": 365.0,
                "available_days_last_30d": 15,
                "available_rate_last_30d": 0.5,
                "avg_minimum_nights_calendar_last_30d": 2.0,
                "avg_maximum_nights_calendar_last_30d": 365.0,
            }
        },
    )

    listing_id: Optional[int] = Field(default=None, ge=0)

    room_type: str
    property_type: str
    neighbourhood_name: str

    accommodates: int = Field(..., ge=0)
    bedrooms: Optional[float] = Field(default=None, ge=0)
    beds: Optional[float] = Field(default=None, ge=0)
    bathrooms: Optional[float] = Field(default=None, ge=0)
    listing_price: Optional[float] = Field(default=None, ge=0)
    minimum_nights: int = Field(..., ge=0)
    maximum_nights: int = Field(..., ge=0)

    instant_bookable: bool
    host_is_superhost: Optional[bool] = None
    host_listing_count: int = Field(..., ge=0)

    total_reviews_before_cutoff: float = Field(..., ge=0)
    unique_reviewers_before_cutoff: float = Field(..., ge=0)
    avg_comment_len_before_cutoff: float = Field(..., ge=0)
    max_comment_len_before_cutoff: float = Field(..., ge=0)
    days_since_last_review: float = Field(..., ge=0)

    available_days_last_90d: int = Field(..., ge=0)
    available_rate_last_90d: float = Field(..., ge=0, le=1)
    avg_minimum_nights_calendar_last_90d: float = Field(..., ge=0)
    avg_maximum_nights_calendar_last_90d: float = Field(..., ge=0)

    available_days_last_30d: int = Field(..., ge=0)
    available_rate_last_30d: float = Field(..., ge=0, le=1)
    avg_minimum_nights_calendar_last_30d: float = Field(..., ge=0)
    avg_maximum_nights_calendar_last_30d: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    listing_id: Optional[int] = None
    prediction: int = Field(..., ge=0, le=1)
    probability_high_demand: float = Field(..., ge=0, le=1)
    model_run_id: str
