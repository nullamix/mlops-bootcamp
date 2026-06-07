@echo off
cd /d "%~dp0"

REM Fill these values from the MLflow credentials sheet.
set MLFLOW_TRACKING_URI=http://185.50.38.163:33014
set MLFLOW_TRACKING_USERNAME=student_amirhossein_sa
set MLFLOW_TRACKING_PASSWORD=your_mlflow_password
set STUDENT_USERNAME=student_amirhossein_sa
set MLFLOW_EXPERIMENT_NAME=qbc12_hw02_student_amirhossein_sa
set MLFLOW_RUN_ID=d97de4e1b1334f1a8996b84d582616f6
set PREDICTION_THRESHOLD=0.5

set PYTHON_EXE=python
if exist ".venv_hw03\Scripts\python.exe" set PYTHON_EXE=.venv_hw03\Scripts\python.exe

%PYTHON_EXE% -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
