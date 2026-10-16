"""HTTP contracts and failures with a controlled recommendation engine."""

import csv
import math

import pytest


def test_health_before_startup(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "not_ready"}


def test_health_when_loaded(client, engine):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", **engine.health_payload.return_value}


def test_readiness_before_startup(client):
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"


def test_readiness_when_loaded(client, engine):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.parametrize("path", ["/recommend", "/warmup"])
def test_inference_requires_loaded_engine(client, path):
    response = client.post(path, json={"user_history": []})
    assert response.status_code == 503


@pytest.mark.parametrize("history", [[], ["N1", "N3"], ["UNKNOWN_NEWS_ID"]])
def test_known_cold_and_unknown_history_forwarded(client, engine, history):
    response = client.post("/recommend", json={"user_history": history})
    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"] == engine.recommend.return_value
    assert body["top_k"] == 10
    assert body["retrieve_k"] == 100
    assert math.isfinite(body["latency_ms"]) and body["latency_ms"] >= 0
    engine.recommend.assert_called_once_with(
        user_history=history, top_k=10, retrieve_k=100, impression_id=None,
    )


@pytest.mark.parametrize("top_k,retrieve_k", [(1, 1), (100, 500)])
def test_valid_boundaries_and_impression_forwarded(client, engine, top_k, retrieve_k):
    response = client.post("/recommend", json={
        "user_history": ["N1"], "top_k": top_k, "retrieve_k": retrieve_k,
        "impression_id": "I1",
    })
    assert response.status_code == 200
    engine.recommend.assert_called_once_with(
        user_history=["N1"], top_k=top_k, retrieve_k=retrieve_k, impression_id="I1",
    )


@pytest.mark.parametrize("payload,field", [
    ({}, "user_history"),
    ({"user_history": None}, "user_history"),
    ({"user_history": "N1 N2"}, "user_history"),
    ({"user_history": [123]}, "user_history"),
    ({"user_history": [], "top_k": 0}, "top_k"),
    ({"user_history": [], "top_k": 101}, "top_k"),
    ({"user_history": [], "top_k": "bad"}, "top_k"),
    ({"user_history": [], "retrieve_k": 0}, "retrieve_k"),
    ({"user_history": [], "retrieve_k": 501}, "retrieve_k"),
    ({"user_history": [], "impression_id": []}, "impression_id"),
])
def test_invalid_requests_rejected_before_inference(client, engine, payload, field):
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422
    assert any(field in error["loc"] for error in response.json()["detail"])
    engine.recommend.assert_not_called()


def test_malformed_json_rejected(client, engine):
    response = client.post("/recommend", content="{", headers={
        "Content-Type": "application/json",
    })
    assert response.status_code == 422
    engine.recommend.assert_not_called()


def test_get_cannot_run_recommendation(client, engine):
    assert client.get("/recommend").status_code == 405
    engine.recommend.assert_not_called()


def test_empty_candidates_are_valid_response(client, engine):
    engine.recommend.return_value = []
    response = client.post("/recommend", json={"user_history": []})
    assert response.status_code == 200
    assert response.json()["recommendations"] == []


def test_optional_metadata_can_be_absent(client, engine):
    engine.recommend.return_value = [{"news_id": "N2", "score": -0.5}]
    response = client.post("/recommend", json={"user_history": []})
    assert response.status_code == 200
    assert response.json()["recommendations"][0]["title"] is None


def test_engine_error_returns_500_without_success_latency_row(api, client, engine):
    engine.recommend.side_effect = RuntimeError("synthetic inference failure")
    response = client.post("/recommend", json={"user_history": []})
    assert response.status_code == 500
    assert not api.LATENCY_LOG.exists()


def test_success_writes_latency_row(api, client, engine):
    response = client.post("/recommend", json={
        "user_history": ["N1", "N3"], "top_k": 5, "retrieve_k": 25,
    })
    assert response.status_code == 200
    with api.LATENCY_LOG.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 1
    assert float(rows[0][1]) >= 0
    assert rows[0][2:] == ["2", "25", "5", "1"]


def test_warmup_uses_first_five_news_items(client, engine):
    response = client.post("/warmup")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "warmed_up"
    assert math.isfinite(body["sample_latency_ms"]) and body["sample_latency_ms"] >= 0
    engine.recommend.assert_called_once_with(
        ["N1", "N2", "N3", "N4", "N5"], top_k=5, retrieve_k=25,
    )


def test_warmup_inference_failure_is_server_error(client, engine):
    engine.recommend.side_effect = RuntimeError("synthetic warmup failure")
    assert client.post("/warmup").status_code == 500


def test_openapi_documents_serving_contract(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert {"/health", "/ready", "/recommend", "/warmup"} <= schema["paths"].keys()
    request = schema["components"]["schemas"]["RecommendRequest"]
    assert "user_history" in request["required"]
    assert request["properties"]["top_k"]["maximum"] == 100
    assert request["properties"]["retrieve_k"]["maximum"] == 500
