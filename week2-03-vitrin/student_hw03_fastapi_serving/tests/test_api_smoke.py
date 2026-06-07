import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import config
from app.main import app, model_service
from app.model_loader import ModelService
from app.predictor import predict_records, records_to_dataframe
from app.schemas import ListingFeatures

PROJECT_DIR = Path(__file__).resolve().parents[1]


class FakeModel:
    classes_ = np.array([0, 1])
    feature_names_in_ = np.array(
        [*config.EXPECTED_FEATURE_COLUMNS, "has_reviews_before_cutoff"]
    )

    def predict_proba(self, features):
        assert list(features.columns) == list(self.feature_names_in_)
        assert features["has_reviews_before_cutoff"].tolist() == [1] * len(features)
        probabilities = np.array([0.2, 0.8][: len(features)])
        return np.column_stack([1 - probabilities, probabilities])


def load_payload(filename: str) -> dict:
    return json.loads((PROJECT_DIR / "data" / filename).read_text(encoding="utf-8"))


def test_root_endpoint():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()


def test_health_and_model_info_endpoints():
    model_service.state.model = FakeModel()
    model_service.state.loaded = True
    model_service.state.error = None
    model_service.state.run_id = "selected-run"
    model_service.state.run_name = "v5_random_forest"
    client = TestClient(app)

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "ok",
        "model_loaded": True,
        "error": None,
    }

    info_response = client.get("/model-info")
    assert info_response.status_code == 200
    assert info_response.json()["run_id"] == "selected-run"
    assert info_response.json()["model_loaded"] is True


def test_records_use_exact_api_feature_order():
    record = ListingFeatures(**load_payload("valid_predict_request.json"))
    dataframe = records_to_dataframe([record])
    assert list(dataframe.columns) == config.EXPECTED_FEATURE_COLUMNS


def test_records_reject_leakage_fields_defensively():
    payload = load_payload("valid_predict_request.json")
    payload["high_demand_proxy"] = 1

    with pytest.raises(HTTPException) as exc_info:
        records_to_dataframe([payload])

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["forbidden_fields"] == ["high_demand_proxy"]


def test_predict_uses_probability_threshold_and_internal_derived_feature():
    records = [
        ListingFeatures(**record)
        for record in load_payload("valid_batch_request.json")["records"]
    ]
    predictions = predict_records(FakeModel(), records)

    assert [item.prediction for item in predictions] == [0, 1]
    assert [item.probability for item in predictions] == [0.2, 0.8]
    assert predictions[1].prediction_label == config.POSITIVE_LABEL


def test_predict_endpoint_and_validation_errors():
    model_service.state.model = FakeModel()
    model_service.state.loaded = True
    model_service.state.error = None
    client = TestClient(app)

    success = client.post("/predict", json=load_payload("valid_predict_request.json"))
    assert success.status_code == 200
    assert success.json()["probability"] == 0.2

    for filename in [
        "bad_request_missing_field.json",
        "bad_request_wrong_type.json",
        "bad_request_leakage_field.json",
    ]:
        response = client.post("/predict", json=load_payload(filename))
        assert response.status_code == 422


def test_predict_batch_endpoint():
    model_service.state.model = FakeModel()
    model_service.state.loaded = True
    model_service.state.error = None
    client = TestClient(app)

    response = client.post(
        "/predict-batch", json=load_payload("valid_batch_request.json")
    )
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert [item["prediction"] for item in response.json()["predictions"]] == [0, 1]


def test_auto_selection_prefers_serving_tag_over_metric_score(monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_EXPERIMENT_NAME", "test-experiment")

    def run(run_id, f1, selected=False):
        tags = {
            "leakage_status": "clean",
            "model_family": "random_forest",
        }
        if selected:
            tags["selected_for_serving"] = "true"
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id),
            data=SimpleNamespace(
                tags=tags,
                metrics={
                    "f1": f1,
                    "roc_auc": 0.9,
                    "precision": 0.9,
                    "recall": 0.9,
                },
            ),
        )

    client = SimpleNamespace(
        get_experiment_by_name=lambda _: SimpleNamespace(experiment_id="31"),
        search_runs=lambda **_: [
            run("higher-unselected", 0.99),
            run("selected", 0.91, selected=True),
        ],
    )

    assert ModelService._select_run_id(client) == "selected"


def test_local_model_fallback_loads_after_mlflow_failure(monkeypatch):
    service = ModelService()
    monkeypatch.setattr(
        service,
        "_load_from_mlflow",
        lambda: (_ for _ in ()).throw(RuntimeError("MLflow unavailable")),
    )

    service.load()

    assert service.state.loaded is True
    assert service.state.run_id == config.SELECTED_RUN_ID
    assert service.state.tags["model_source"] == "local_fallback"
    assert "MLflow unavailable" in service.state.tags["mlflow_load_error"]
