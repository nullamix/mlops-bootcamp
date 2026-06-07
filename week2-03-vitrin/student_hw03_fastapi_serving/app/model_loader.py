from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from . import config


@dataclass
class LoadedModelState:
    model: Any = None
    loaded: bool = False
    error: Optional[str] = None
    model_uri: Optional[str] = None
    run_id: Optional[str] = None
    run_name: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)


class ModelService:
    """Load the selected clean HW02 model without crashing API startup."""

    def __init__(self) -> None:
        self.state = LoadedModelState()

    def load(self) -> None:
        self.state = LoadedModelState()

        try:
            self._load_from_mlflow()
            return
        except Exception as exc:
            mlflow_error = self._format_error(exc)

        try:
            self._load_local_model(config.LOCAL_MODEL_PATH, mlflow_error)
        except Exception as exc:
            local_error = self._format_error(exc)
            errors = []
            if mlflow_error:
                errors.append(f"MLflow load failed: {mlflow_error}")
            errors.append(f"Local fallback failed: {local_error}")
            self.state.error = "; ".join(errors)

    def _load_from_mlflow(self) -> None:
        if config.MLFLOW_TRACKING_USERNAME:
            os.environ["MLFLOW_TRACKING_USERNAME"] = config.MLFLOW_TRACKING_USERNAME
        if config.MLFLOW_TRACKING_PASSWORD:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = config.MLFLOW_TRACKING_PASSWORD

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)

        run_id = config.MLFLOW_RUN_ID or self._select_run_id(client)
        model_uri = f"runs:/{run_id}/model"
        run = client.get_run(run_id)
        model = mlflow.sklearn.load_model(model_uri)

        self.state = LoadedModelState(
            model=model,
            loaded=True,
            model_uri=model_uri,
            run_id=run_id,
            run_name=run.data.tags.get("mlflow.runName"),
            metrics=dict(run.data.metrics),
            params=dict(run.data.params),
            tags={**dict(run.data.tags), "model_source": "mlflow"},
        )

    def _load_local_model(
        self, model_directory: Path, mlflow_error: Optional[str]
    ) -> None:
        model_file = model_directory / "model.pkl"
        if not model_file.is_file():
            raise FileNotFoundError(f"Local model file not found: {model_file}")

        with model_file.open("rb") as file:
            model = pickle.load(file)

        tags = {
            "leakage_status": "clean",
            "model_family": "random_forest",
            "selected_for_serving": "true",
            "model_source": "local_fallback",
        }
        if mlflow_error:
            tags["mlflow_load_error"] = mlflow_error

        self.state = LoadedModelState(
            model=model,
            loaded=True,
            model_uri=model_directory.resolve().as_uri(),
            run_id=config.SELECTED_RUN_ID,
            run_name="v5_random_forest",
            params={"threshold": str(config.PREDICTION_THRESHOLD)},
            tags=tags,
        )

    @staticmethod
    def _format_error(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _select_run_id(client: Any) -> str:
        experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            raise RuntimeError(
                f"MLflow experiment {config.MLFLOW_EXPERIMENT_NAME!r} was not found."
            )

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            max_results=1000,
        )
        eligible_runs = [
            run
            for run in runs
            if run.data.tags.get("leakage_status", "").lower() == "clean"
            and run.data.tags.get("model_family", "").lower() != "dummy"
        ]
        if not eligible_runs:
            raise RuntimeError(
                f"No clean non-dummy runs found in {config.MLFLOW_EXPERIMENT_NAME!r}."
            )

        selected_runs = [
            run
            for run in eligible_runs
            if ModelService._is_true_tag(
                run.data.tags.get("selected_for_serving")
            ) or ModelService._is_true_tag(
                run.data.tags.get("production_candidate")
            )
        ]
        candidates = selected_runs or eligible_runs
        best_run = max(candidates, key=ModelService._run_score)
        return best_run.info.run_id

    @staticmethod
    def _is_true_tag(value: Optional[str]) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _run_score(run: Any) -> tuple[float, float, float, float]:
        metrics = run.data.metrics

        def metric(name: str) -> float:
            value = metrics.get(name)
            return float(value) if value is not None else float("-inf")

        return (
            metric("f1"),
            metric("roc_auc"),
            metric("precision"),
            metric("recall"),
        )

    def require_model(self):
        if not self.state.loaded or self.state.model is None:
            raise RuntimeError(self.state.error or "Model is not loaded.")
        return self.state.model

    def model_info(self) -> dict:
        return {
            "model_loaded": self.state.loaded,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "experiment_name": config.MLFLOW_EXPERIMENT_NAME,
            "model_uri": self.state.model_uri,
            "run_id": self.state.run_id,
            "run_name": self.state.run_name,
            "dataset_version": config.DATASET_VERSION,
            "target": config.TARGET_NAME,
            "threshold": config.PREDICTION_THRESHOLD,
            "metrics": self.state.metrics,
            "params": self.state.params,
            "tags": self.state.tags,
            "error": self.state.error,
        }
