"""Session and UI adapter contracts, without external HTTP requests."""

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from demo.server import COOKIE, SessionStore, create_app


@pytest.fixture
def demo(tmp_path):
    news = tmp_path / "news.tsv"
    news.write_text("".join(
        f"N{i}\ttech\tai\tArticle {i}\tAbstract {i}\thttps://example.test/{i}\t[]\t[]\n"
        for i in range(1, 9)
    ), encoding="utf-8")
    calls, mode = [], {"fail": False}

    def upstream(request):
        assert str(request.url) == "http://recommender.test/recommend"
        body = json.loads(request.content)
        calls.append(body)
        if mode["fail"]:
            return httpx.Response(503, json={"detail": "Not ready"})
        unseen = [f"N{i}" for i in range(1, 9) if f"N{i}" not in body["user_history"]]
        return httpx.Response(200, json={
            "recommendations": [{"news_id": n, "score": 0.5} for n in unseen[:4]],
            "latency_ms": 12.3,
        })

    app = create_app(news, tmp_path / "sessions.db", "http://recommender.test",
                     transport=httpx.MockTransport(upstream))
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, app=app, calls=calls, mode=mode)


def test_demo_page_assets_and_catalog(demo):
    assert demo.client.get("/").status_code == 200
    for asset in ["app.js", "styles.css"]:
        assert demo.client.get("/static/" + asset).status_code == 200
    catalog = demo.client.get("/api/catalog").json()
    assert len(catalog["articles"]) == 4
    assert catalog["articles"][0]["abstract"] == "Abstract 1"
    assert demo.calls == []


def test_new_session_sets_private_cookie_and_calls_real_contract(demo):
    response = demo.client.get("/api/session")
    body = response.json()
    assert body["read_count"] == 0
    assert len(body["recommendations"]) == 4
    assert body["api_connected"] is True
    assert "HttpOnly" in response.headers["set-cookie"]
    assert response.headers["cache-control"] == "no-store"
    assert demo.calls[-1] == {"user_history": [], "top_k": 4, "retrieve_k": 100}


def test_read_saves_history_refreshes_four_and_survives_reload(demo):
    before = demo.client.get("/api/session").json()
    after = demo.client.post("/api/read", json={"news_id": "N1"}).json()
    assert after["session"] == before["session"]
    assert after["read_count"] == 1
    assert after["history"][0]["news_id"] == "N1"
    assert len(after["recommendations"]) == 4
    assert "N1" not in [r["news_id"] for r in after["recommendations"]]
    assert demo.calls[-1]["user_history"] == ["N1"]
    assert demo.client.get("/api/session").json()["read_count"] == 1


def test_repeat_read_is_idempotent(demo):
    demo.client.post("/api/read", json={"news_id": "N1"})
    result = demo.client.post("/api/read", json={"news_id": "N1"}).json()
    assert result["read_count"] == 1
    assert demo.calls[-1]["user_history"] == ["N1"]


def test_browser_sessions_are_isolated(demo):
    demo.client.post("/api/read", json={"news_id": "N1"})
    first_cookie = demo.client.cookies.get(COOKIE)
    demo.client.cookies.clear()
    second = demo.client.get("/api/session").json()
    assert second["read_count"] == 0
    assert demo.client.cookies.get(COOKIE) != first_cookie
    demo.client.cookies.clear()
    demo.client.cookies.set(COOKIE, first_cookie)
    assert demo.client.get("/api/session").json()["read_count"] == 1


def test_new_reader_and_reset_history(demo):
    first = demo.client.post("/api/read", json={"news_id": "N1"}).json()
    reset = demo.client.post("/api/session/reset").json()
    assert reset["session"] == first["session"]
    assert reset["read_count"] == 0
    new = demo.client.post("/api/session/new").json()
    assert new["session"] != first["session"]
    assert new["read_count"] == 0


def test_upstream_failure_keeps_read_history_without_faking_recommendations(demo):
    demo.mode["fail"] = True
    response = demo.client.post("/api/read", json={"news_id": "N1"})
    assert response.status_code == 200
    body = response.json()
    assert body["read_count"] == 1
    assert body["api_connected"] is False
    assert body["recommendations"] == []
    assert body["error"]
    demo.mode["fail"] = False
    assert demo.client.get("/api/session").json()["api_connected"] is True


def test_unknown_article_and_invalid_payload_are_rejected(demo):
    assert demo.client.post("/api/read", json={"news_id": "BAD"}).status_code == 404
    assert demo.client.post("/api/read", json={}).status_code == 422
    assert demo.calls == []


def test_cross_origin_mutation_is_rejected(demo):
    response = demo.client.post("/api/read", json={"news_id": "N1"},
                                headers={"Origin": "https://other.test"})
    assert response.status_code == 403
    assert demo.calls == []


def test_session_store_persists_across_server_instances(tmp_path):
    path = tmp_path / "sessions.db"
    first = SessionStore(path)
    token, created = first.session(None)
    assert created
    first.read(token, "N1")
    second = SessionStore(path)
    assert second.session(token) == (token, False)
    assert second.history(token) == ["N1"]
