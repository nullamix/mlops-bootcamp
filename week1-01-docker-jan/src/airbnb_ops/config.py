from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    listings_path: Path = Path("data/raw/listings_sample.csv")
    segments_path: Path = Path("data/raw/neighbourhood_segments.csv")
    output_path: Path = Path("data/processed/airbnb_neighbourhood_summary.csv")
    report_path: Path = Path("reports/hw01_a_run_report.md")
