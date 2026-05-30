from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

from airbnb_ops.config import PipelineConfig
from airbnb_ops.extract import read_csv_checked
from airbnb_ops.pii import handle_pii
from airbnb_ops.transform import build_neighbourhood_summary
from airbnb_ops.validate import validate_summary

app = typer.Typer(help="Airbnb ops pipeline", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Airbnb ops command group."""
    pass


def _write_report(summary: pd.DataFrame, config: PipelineConfig) -> None:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = f"""# HW01-A Run Report

Generated at: `{generated_at}`

## Inputs

- Listings: `{config.listings_path}`
- Segments: `{config.segments_path}`

## Outputs

- Summary CSV: `{config.output_path}`
- Report: `{config.report_path}`

## Validation

Status: `passed`

## Result Shape

- Rows: `{len(summary)}`
- Columns: `{len(summary.columns)}`

## Neighbourhood Summary Preview

```csv\n{summary.to_csv(index=False).strip()}\n```
"""
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(report, encoding="utf-8")


@app.command()
def run(
    listings_path: Path = typer.Option(
        Path("data/raw/listings_sample.csv"), help="Path to raw listings CSV"
    ),
    segments_path: Path = typer.Option(
        Path("data/raw/neighbourhood_segments.csv"), help="Path to segments CSV"
    ),
    output_path: Path = typer.Option(
        Path("data/processed/airbnb_neighbourhood_summary.csv"), help="Path to output CSV"
    ),
    report_path: Path = typer.Option(
        Path("reports/hw01_a_run_report.md"), help="Path to markdown report"
    ),
) -> None:
    """Run the Airbnb neighbourhood summary pipeline."""
    config = PipelineConfig(
        listings_path=listings_path,
        segments_path=segments_path,
        output_path=output_path,
        report_path=report_path,
    )

    listings = read_csv_checked(config.listings_path)
    segments = read_csv_checked(config.segments_path)

    safe_listings = handle_pii(listings)
    summary = build_neighbourhood_summary(safe_listings, segments)
    validate_summary(summary)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.output_path, index=False)
    _write_report(summary, config)

    typer.echo(f"Wrote {config.output_path}")
    typer.echo(f"Wrote {config.report_path}")


if __name__ == "__main__":
    app()
