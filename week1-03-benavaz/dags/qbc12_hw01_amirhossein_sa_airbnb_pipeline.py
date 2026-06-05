from datetime import timedelta
from pathlib import Path
from typing import Any
import pendulum
from airflow.sdk import dag, task
from airflow.models import Variable
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


DAG_ID = "qbc12_hw01_amirhossein_sa_airbnb_pipeline"
STUDENT_SCHEMA = "student_amirhossein_sa"
MATERIALIZED_VIEW = "mv_airbnb_neighbourhood_summary"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


def _airflow_variable(*names: str, default: str | None = None) -> str:
    print("name:", names)
    print("default:", default)
    for name in names:
        value = Variable.get(name, default_var=None)
        if value not in (None, ""):
            return value
    if default is not None:
        return default
    raise KeyError(f"Missing required Airflow Variable. Tried: {', '.join(names)}")


def make_engine(config: dict[str, Any]):
    url = URL.create(
        "postgresql+psycopg2",
        username=config["db_user"],
        password=config["db_password"],
        host=config["db_host"],
        port=int(config["db_port"]),
        database=config["db_name"],
    )
    return create_engine(url, pool_pre_ping=True)


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _object_name(config: dict[str, Any]) -> str:
    return f"{_quote_ident(config['schema'])}.{_quote_ident(MATERIALIZED_VIEW)}"


def _write_report(filename: str, body: str) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / filename
    path.write_text(body, encoding="utf-8")
    return str(path)


@dag(
    dag_id=DAG_ID,
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args={
        "owner": "amirhossein_sa",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["qbc12", "hw01", "airbnb"],
    description="Refresh and validate the HW01 Airbnb neighbourhood materialized view.",
)
def airbnb_pipeline():
    @task
    def read_config() -> dict[str, Any]:
        return {
            "db_host": _airflow_variable("QBC12_DB_HOST", "qbc12_db_host"),
            "db_port": _airflow_variable("QBC12_DB_PORT", "qbc12_db_port", default="32112"),
            "db_name": _airflow_variable("QBC12_DB_NAME", "qbc12_db_name", default="qbc12_airbnb"),
            "db_user": _airflow_variable("QBC12_DB_USER", "qbc12_db_user"),
            "db_password": _airflow_variable("QBC12_DB_PASSWORD", "qbc12_db_password"),
            "schema": STUDENT_SCHEMA,
            "view_name": MATERIALIZED_VIEW,
        }

    @task
    def refresh_summary(config: dict[str, Any]) -> dict[str, Any]:
        schema = _quote_ident(config["schema"])
        object_name = _object_name(config)

        refresh_sql = f"""
        create schema if not exists {schema};

        drop materialized view if exists {object_name};

        create materialized view {object_name} as
        with listing_clean as (
            select
                id as listing_id,
                coalesce(nullif(btrim(neighbourhood), ''), 'Unknown') as neighbourhood,
                nullif(regexp_replace(price::text, '[^0-9.-]', '', 'g'), '')::numeric as listing_price,
                nullif(regexp_replace(minimum_nights::text, '[^0-9.-]', '', 'g'), '')::numeric as minimum_nights
            from core.listing
        ),
        calendar_clean as (
            select
                listing_id,
                date,
                nullif(regexp_replace(price::text, '[^0-9.-]', '', 'g'), '')::numeric as calendar_price,
                case
                    when lower(available::text) in ('true', 't', '1', 'yes', 'y') then 1.0
                    else 0.0
                end as available_num
            from core.calendar_day
            where date >= current_date
              and date < current_date + interval '365 days'
        ),
        calendar_agg as (
            select
                listing_id,
                avg(calendar_price) filter (
                    where date < current_date + interval '30 days'
                ) as avg_calendar_price_30,
                avg(available_num) filter (
                    where date < current_date + interval '30 days'
                ) as availability_30_rate,
                avg(available_num) as availability_365_rate
            from calendar_clean
            group by listing_id
        ),
        review_counts as (
            select listing_id, count(*)::bigint as total_reviews
            from core.review
            group by listing_id
        )
        select
            l.neighbourhood,
            count(*)::bigint as num_listings,
            round(avg(coalesce(l.listing_price, c.avg_calendar_price_30))::numeric, 2) as avg_price,
            round(
                percentile_cont(0.5) within group (
                    order by coalesce(l.listing_price, c.avg_calendar_price_30)
                )::numeric,
                2
            ) as median_price,
            round(avg(l.minimum_nights)::numeric, 2) as avg_minimum_nights,
            coalesce(sum(r.total_reviews), 0)::bigint as total_reviews,
            round((coalesce(sum(r.total_reviews), 0)::numeric / nullif(count(*), 0)), 2) as reviews_per_listing,
            round(avg(coalesce(c.availability_30_rate, 0))::numeric, 4) as availability_30_rate,
            round(avg(coalesce(c.availability_365_rate, 0))::numeric, 4) as availability_365_rate,
            now() as refreshed_at
        from listing_clean l
        left join calendar_agg c on c.listing_id = l.listing_id
        left join review_counts r on r.listing_id = l.listing_id
        group by l.neighbourhood;

        create unique index if not exists mv_airbnb_neighbourhood_summary_neighbourhood_uidx
            on {object_name} (neighbourhood);

        create index if not exists mv_airbnb_neighbourhood_summary_listings_idx
            on {object_name} (num_listings desc);

        create index if not exists mv_airbnb_neighbourhood_summary_price_idx
            on {object_name} (avg_price);
        """

        engine = make_engine(config)
        with engine.begin() as conn:
            conn.execute(text("set statement_timeout = '120s'"))
            for statement in [stmt.strip() for stmt in refresh_sql.split(";") if stmt.strip()]:
                conn.execute(text(statement))
            row = conn.execute(
                text(
                    f"""
                    select count(*) as row_count, max(refreshed_at) as refreshed_at
                    from {object_name}
                    """
                )
            ).mappings().one()

        return {
            "schema": config["schema"],
            "object": f"{config['schema']}.{MATERIALIZED_VIEW}",
            "row_count": int(row["row_count"]),
            "refreshed_at": row["refreshed_at"].isoformat() if row["refreshed_at"] else None,
        }

    @task
    def validate_summary(config: dict[str, Any], refresh_result: dict[str, Any]) -> dict[str, Any]:
        object_name = _object_name(config)
        validation_sql = f"""
        select
            count(*)::int as row_count,
            count(*) filter (
                where neighbourhood is null or btrim(neighbourhood) = ''
            )::int as null_neighbourhoods,
            count(*) filter (
                where avg_price is null
                   or median_price is null
                   or avg_price <= 0
                   or median_price <= 0
            )::int as bad_prices,
            count(*) filter (
                where availability_30_rate is null
                   or availability_365_rate is null
                   or availability_30_rate < 0
                   or availability_30_rate > 1
                   or availability_365_rate < 0
                   or availability_365_rate > 1
            )::int as bad_availability
        from {object_name};
        """

        engine = make_engine(config)
        with engine.begin() as conn:
            checks = dict(conn.execute(text(validation_sql)).mappings().one())

        checks["passed"] = (
            checks["row_count"] > 0 and checks["null_neighbourhoods"] == 0 and checks["bad_prices"] == 0 and checks["bad_availability"] == 0
        )
        checks["refreshed_object"] = refresh_result["object"]
        checks["refreshed_at"] = refresh_result["refreshed_at"]
        return checks

    @task.branch
    def choose_report_path(validation_result: dict[str, Any]) -> str:
        if validation_result["passed"]:
            return "write_success_report"
        return "write_failure_report"

    @task
    def write_success_report(refresh_result: dict[str, Any], validation_result: dict[str, Any]) -> str:
        body = f"""# HW01-C Airflow Run Report

- DAG id: `{DAG_ID}`
- Refreshed object: `{refresh_result['object']}`
- Refreshed at: `{refresh_result['refreshed_at']}`
- Row count: `{validation_result['row_count']}`
- Validation result: passed
- Checks:
  - null_neighbourhoods: `{validation_result['null_neighbourhoods']}`
  - bad_prices: `{validation_result['bad_prices']}`
  - bad_availability: `{validation_result['bad_availability']}`
- Screenshot paths:
  - `screenshots/airflow_dag_graph.png`
  - `screenshots/airflow_success_run.png`
"""
        return _write_report("hw01_c_airflow_success.md", body)

    @task
    def write_failure_report(refresh_result: dict[str, Any], validation_result: dict[str, Any]) -> str:
        body = f"""# HW01-C Airflow Failure Report

- DAG id: `{DAG_ID}`
- Refreshed object: `{refresh_result['object']}`
- Refreshed at: `{refresh_result['refreshed_at']}`
- Validation result: failed
- Checks:
  - row_count: `{validation_result['row_count']}`
  - null_neighbourhoods: `{validation_result['null_neighbourhoods']}`
  - bad_prices: `{validation_result['bad_prices']}`
  - bad_availability: `{validation_result['bad_availability']}`
"""
        path = _write_report("hw01_c_airflow_failure.md", body)
        raise ValueError(f"Validation failed. Failure report written to {path}")

    config = read_config()
    refresh_result = refresh_summary(config)  # type: ignore
    validation_result = validate_summary(config, refresh_result)  # type: ignore
    report_path = choose_report_path(validation_result)  # type: ignore
    success_report = write_success_report(refresh_result, validation_result)  # type: ignore
    failure_report = write_failure_report(refresh_result, validation_result)  # type: ignore

    report_path >> [success_report, failure_report]


airbnb_pipeline()
