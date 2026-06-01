# HW01-C Airflow Scheduled Pipeline

- DAG id: `qbc12_hw01_amirhossein_sa_airbnb_pipeline`
- Airflow URL: shared QBC12 Airflow URL from the homework page
- Successful run timestamp: pending shared Airflow trigger
- Refreshed object name: `student_amirhossein_sa.mv_airbnb_neighbourhood_summary`
- Validation result: the DAG validates row_count > 0, null_neighbourhoods == 0, bad_prices == 0, and bad_availability == 0
- Screenshot paths:
  - `screenshots/airflow_dag_graph.png`
  - `screenshots/airflow_success_run.png`

## Notes

The DAG file was generated from this notebook and syntax-checked locally. After triggering it in shared Airflow, update the timestamp above and place the required screenshots at the listed paths.

Report generated locally at: 2026-06-01T13:12:08.804448+00:00
