#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://185.50.38.163:33014}"
export MLFLOW_TRACKING_USERNAME="${MLFLOW_TRACKING_USERNAME:-student_amirhossein_sa}"
export STUDENT_USERNAME="${STUDENT_USERNAME:-student_amirhossein_sa}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-qbc12_hw02_student_amirhossein_sa}"
export MLFLOW_RUN_ID="${MLFLOW_RUN_ID:-d97de4e1b1334f1a8996b84d582616f6}"
export PREDICTION_THRESHOLD="${PREDICTION_THRESHOLD:-0.5}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x "../../.venv/bin/python" ]]; then
  PYTHON_BIN="../../.venv/bin/python"
else
  PYTHON_BIN="/usr/bin/python"
fi

"$PYTHON_BIN" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
