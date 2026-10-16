"""Loading, serving logic, and lifecycle using only tiny local fixtures."""

from dataclasses import replace
import json
import pickle
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
@pytest.mark.parametrize("history", [[], ["N1"], ["UNKNOWN"]])
def test_real_startup_and_inference(api, artifact_settings, history):
    with TestClient(api.app) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/health").json() == {
            "status": "ok", "faiss_vectors": 4, "num_embeddings": 4,
            "num_news": 4, "xgb_model_loaded": True,
        }
        response = client.post("/recommend", json={
            "user_history": history, "top_k": 2, "retrieve_k": 4,
            "impression_id": "I1",
        })
        assert response.status_code == 200
        recs = response.json()["recommendations"]
        assert len(recs) == 2
        ids = [r["news_id"] for r in recs]
        assert len(set(ids)) == len(ids)
        assert not set(ids).intersection(history)
        assert set(ids) <= {"N1", "N2", "N3", "N4"}
        assert all(r["title"] and np.isfinite(r["score"]) for r in recs)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)
        assert client.post("/warmup").status_code == 200


@pytest.mark.integration
def test_startup_without_optional_ctr(api, artifact_settings, monkeypatch):
    monkeypatch.setattr(api, "SETTINGS", replace(artifact_settings, CTR_MAP_PATH=""))
    with TestClient(api.app) as client:
        assert client.get("/ready").status_code == 200
        assert api.ENGINE.ctr_map == {}


@pytest.mark.parametrize("field,message", [
    ("NEWS_METADATA_PATH", "News metadata file not found"),
    ("FAISS_INDEX_PATH", "FAISS index not found"),
    ("EMBEDDINGS_MAP_PATH", "Embeddings map not found"),
    ("XGB_MODEL_PATH", "XGBoost model not found"),
])
def test_missing_required_artifact_fails_startup(
    api, artifact_settings, monkeypatch, tmp_path, field, message,
):
    monkeypatch.setattr(api, "SETTINGS", replace(
        artifact_settings, **{field: str(tmp_path / "missing")},
    ))
    with pytest.raises(FileNotFoundError, match=message):
        with TestClient(api.app):
            pytest.fail("Startup must not succeed with missing required artifacts")
    assert api.ENGINE is None


@pytest.mark.parametrize("suffix,separator", [(".tsv", "\t"), (".csv", ",")])
def test_news_metadata_loading(api, tmp_path, suffix, separator):
    path = tmp_path / ("news" + suffix)
    path.write_text(separator.join([
        "N1", "tech", "ai", "An article", "Abstract", "url", "[]", "[]",
    ]) + "\n", encoding="utf-8")
    store = api.NewsStore(str(path))
    assert store.get_news_row("N1") == {
        "news_id": "N1", "title": "An article", "category": "tech", "subcategory": "ai",
    }
    assert store.get_news_row("UNKNOWN")["title"] is None


def test_unsupported_news_format_rejected(api, tmp_path):
    path = tmp_path / "news.txt"
    path.write_text("N1", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported news file format"):
        api.NewsStore(str(path))


def test_missing_metadata_columns_rejected(api, tmp_path, monkeypatch):
    path = tmp_path / "news.parquet"
    path.touch()
    monkeypatch.setattr(api.pd, "read_parquet", lambda _: api.pd.DataFrame({"news_id": ["N1"]}))
    with pytest.raises(ValueError, match="missing required columns"):
        api.NewsStore(str(path))


@pytest.mark.parametrize("contents,expected", [
    ('{"I1": {"N2": 0.25}}', {"I1": {"N2": 0.25}}),
    ("[]", {}), ("null", {}), ("not-json", {}),
])
def test_optional_ctr_map_loading(api, tmp_path, contents, expected):
    path = tmp_path / "ctr.json"
    path.write_text(contents, encoding="utf-8")
    assert api.load_ctr_map(str(path)) == expected


def test_absent_ctr_map_is_allowed(api, tmp_path):
    assert api.load_ctr_map("") == {}
    assert api.load_ctr_map(str(tmp_path / "missing.json")) == {}


def test_invalid_embedding_map_rejected(api, artifacts, tmp_path):
    path = tmp_path / "invalid.pkl"
    path.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="must contain a dict"):
        api.HybridRetriever.load(str(artifacts.index), str(path))


@pytest.fixture
def retriever(api, artifacts):
    return api.HybridRetriever.load(str(artifacts.index), str(artifacts.embeddings))


def test_cold_retrieval_has_bounded_deterministic_fallback(retriever):
    assert retriever.retrieve_candidates(np.zeros(3, dtype=np.float32), 2) == ["N1", "N2"]


def test_dense_retrieval_maps_indices_and_ignores_faiss_padding(retriever):
    results = retriever.retrieve_candidates(np.array([1, 0, 0], dtype=np.float32), 10)
    assert results[:2] == ["N1", "N2"]
    assert len(results) == 4
    assert len(set(results)) == 4


def test_user_vector_ignores_unknown_ids_and_weights_recent_history(retriever):
    vector = retriever.generate_user_vector(["N1", "UNKNOWN", "N3"])
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) == pytest.approx(1)
    assert vector[1] > vector[0]
    np.testing.assert_array_equal(
        retriever.generate_user_vector(["UNKNOWN"]), np.zeros(3, dtype=np.float32),
    )


@pytest.fixture
def serving_engine(api, retriever, artifacts):
    model = Mock()
    model.predict.side_effect = lambda matrix: np.arange(matrix.num_row(), dtype=np.float32)
    return api.RecommendationEngine(
        retriever, api.NewsStore(str(artifacts.news)), model,
        ctr_map={"I1": {"N2": 0.25}},
    )


def test_reranking_orders_scores_and_excludes_history(serving_engine):
    recs = serving_engine.recommend(["N1"], top_k=2, retrieve_k=4)
    assert len(recs) == 2
    assert "N1" not in [r["news_id"] for r in recs]
    assert [r["score"] for r in recs] == [2.0, 1.0]
    matrix = serving_engine.model.predict.call_args.args[0]
    assert matrix.num_col() == 14
    assert matrix.feature_names[-1] == "global_ctr"


def test_empty_retrieval_skips_ranking(serving_engine, monkeypatch):
    monkeypatch.setattr(serving_engine.retriever, "retrieve_candidates", lambda *a, **k: [])
    assert serving_engine.recommend(["N1"]) == []
    serving_engine.model.predict.assert_not_called()


def test_all_candidates_seen_returns_empty_without_ranking(serving_engine):
    assert serving_engine.recommend(["N1", "N2", "N3", "N4"], retrieve_k=4) == []
    serving_engine.model.predict.assert_not_called()


def test_fewer_unseen_candidates_than_top_k_is_allowed(serving_engine):
    recs = serving_engine.recommend(["N1", "N2", "N3"], top_k=10, retrieve_k=4)
    assert [r["news_id"] for r in recs] == ["N4"]


def test_features_include_affinities_ctr_and_similarity(serving_engine):
    profile = serving_engine._get_user_profile(["N1", "N3"])
    row = serving_engine._feature_row("N2", 1, 4, profile, "I1")
    assert row["user_hist_len"] == 2
    assert row["user_cat_affinity"] == pytest.approx(0.5)
    assert row["user_subcat_affinity"] == pytest.approx(0.5)
    assert row["user_history_entropy"] == pytest.approx(1)
    assert row["dense_cosine_sim"] == pytest.approx(1.4 / np.sqrt(2))
    assert row["normalized_imp_position"] == pytest.approx(0.25)
    assert row["global_ctr"] == pytest.approx(0.25)
    assert serving_engine._feature_row("N2", 0, 4, profile, "UNKNOWN")["global_ctr"] == 0


def test_cold_features_are_finite(serving_engine):
    profile = serving_engine._get_user_profile([])
    row = serving_engine._feature_row("N2", 0, 4, profile)
    assert all(np.isfinite(v) for k, v in row.items() if k != "candidate_id")
    assert row["dense_cosine_sim"] == 0
    assert row["user_hist_len"] == 0
    assert row["global_ctr"] == 0
