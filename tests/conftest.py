"""Isolated serving fixtures: no production files, downloads, or AWS calls."""

import importlib.util
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import faiss
import numpy as np
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient


FEATURES = [
    "dense_cosine_sim", "max_cosine_sim", "recency_weighted_sim",
    "category_match", "user_cat_affinity", "subcategory_match", "top3_sim",
    "user_subcat_affinity", "is_new_category", "user_hist_len",
    "user_history_entropy", "impression_position", "normalized_imp_position",
    "global_ctr",
]


@pytest.fixture(scope="session")
def api(tmp_path_factory):
    # app.py creates a CSV at import. Import from a temporary working directory
    # rather than touching the developer's real latency_log.csv.
    source = Path(__file__).resolve().parents[1] / "app.py"
    name = "recommender_api_under_test"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.chdir(tmp_path_factory.mktemp("api-import"))
            spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def isolate_state(api, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "ENGINE", None)
    monkeypatch.setattr(api, "STARTUP_TIME", None)
    monkeypatch.setattr(api, "LATENCY_LOG", tmp_path / "latency.csv")


@pytest.fixture
def client(api):
    # Deliberately not a context manager: endpoint unit tests do not run startup.
    # Separate integration tests below use `with TestClient(...)` for startup.
    with_client = TestClient(api.app, raise_server_exceptions=False)
    yield with_client
    with_client.close()


@pytest.fixture
def engine(api, monkeypatch):
    fake = SimpleNamespace(
        recommend=Mock(return_value=[{
            "news_id": "N2", "score": 0.75, "title": "Test article",
            "category": "tech", "subcategory": "ai",
        }]),
        health_payload=Mock(return_value={
            "faiss_vectors": 4, "num_embeddings": 4,
            "num_news": 4, "xgb_model_loaded": True,
        }),
        news_store=SimpleNamespace(df_news=api.pd.DataFrame({
            "news_id": ["N1", "N2", "N3", "N4", "N5", "N6"],
        })),
    )
    monkeypatch.setattr(api, "ENGINE", fake)
    return fake


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-artifacts")
    embeddings = {
        "N1": np.array([1, 0, 0], dtype=np.float32),
        "N2": np.array([0.8, 0.6, 0], dtype=np.float32),
        "N3": np.array([0, 1, 0], dtype=np.float32),
        "N4": np.array([0, 0, 1], dtype=np.float32),
    }
    news = root / "news.tsv"
    news.write_text(
        "N1\ttech\tai\tArticle one\tAbstract\thttps://example.test/1\t[]\t[]\n"
        "N2\ttech\tai\tArticle two\tAbstract\thttps://example.test/2\t[]\t[]\n"
        "N3\tsport\tfootball\tArticle three\tAbstract\thttps://example.test/3\t[]\t[]\n"
        "N4\tnews\tworld\tArticle four\tAbstract\thttps://example.test/4\t[]\t[]\n",
        encoding="utf-8",
    )
    embedding_path = root / "embeddings.pkl"
    embedding_path.write_bytes(pickle.dumps(embeddings))
    index = faiss.IndexFlatIP(3)
    index.add(np.stack(list(embeddings.values())))
    index_path = root / "index.bin"
    faiss.write_index(index, str(index_path))
    rng = np.random.default_rng(42)
    matrix = xgb.DMatrix(
        rng.random((8, len(FEATURES)), dtype=np.float32),
        label=np.arange(8, dtype=np.float32), feature_names=FEATURES,
    )
    model = xgb.train(
        {"objective": "reg:squarederror", "max_depth": 1, "seed": 42,
         "nthread": 1}, matrix, num_boost_round=2,
    )
    model_path = root / "ranker.json"
    model.save_model(str(model_path))
    ctr = root / "ctr.json"
    ctr.write_text(json.dumps({"I1": {"N2": 0.25}}), encoding="utf-8")
    return SimpleNamespace(news=news, embeddings=embedding_path,
                           index=index_path, model=model_path, ctr=ctr)


@pytest.fixture
def artifact_settings(api, artifacts, monkeypatch):
    settings = api.Settings(
        NEWS_METADATA_PATH=str(artifacts.news),
        FAISS_INDEX_PATH=str(artifacts.index),
        EMBEDDINGS_MAP_PATH=str(artifacts.embeddings),
        XGB_MODEL_PATH=str(artifacts.model),
        CTR_MAP_PATH=str(artifacts.ctr),
    )
    monkeypatch.setattr(api, "SETTINGS", settings)
    return settings
