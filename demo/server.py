"""Run with: uvicorn demo.server:app --host 127.0.0.1 --port 8080."""

from contextlib import asynccontextmanager, closing
import csv
import logging
import math
import os
from pathlib import Path
import secrets
import sqlite3
import time
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
COOKIE = "next_read_session"
SESSION_SECONDS = 7 * 24 * 60 * 60
logger = logging.getLogger("reading-demo")


def load_catalog(path):
    """Only public metadata and abstracts; no model or embeddings loaded here."""
    catalog = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) < 5 or not row[0] or not row[3]:
                continue
            url = row[5] if len(row) > 5 else ""
            if urlsplit(url).scheme not in {"http", "https"}:
                url = ""
            catalog[row[0]] = {
                "news_id": row[0], "category": row[1], "subcategory": row[2],
                "title": row[3], "abstract": row[4], "url": url,
            }
    return catalog


class SessionStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, expires REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reads (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    news_id TEXT NOT NULL,
                    UNIQUE(session_id, news_id)
                );
            """)

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def session(self, token):
        now = time.time()
        with closing(self.connect()) as connection, connection:
            connection.execute("DELETE FROM sessions WHERE expires <= ?", (now,))
            exists = token and connection.execute(
                "SELECT 1 FROM sessions WHERE id=?", (token,),
            ).fetchone()
            if exists:
                return token, False
            token = secrets.token_urlsafe(32)
            connection.execute("INSERT INTO sessions VALUES (?, ?)",
                               (token, now + SESSION_SECONDS))
            return token, True

    def read(self, token, news_id):
        with closing(self.connect()) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO reads(session_id, news_id) VALUES (?, ?)",
                (token, news_id),
            )

    def history(self, token):
        with closing(self.connect()) as connection:
            return [row[0] for row in connection.execute(
                "SELECT news_id FROM reads WHERE session_id=? ORDER BY sequence",
                (token,),
            )]

    def clear(self, token):
        with closing(self.connect()) as connection, connection:
            connection.execute("DELETE FROM reads WHERE session_id=?", (token,))


class ReadRequest(BaseModel):
    news_id: str = Field(min_length=1, max_length=100)


def create_app(news_path=None, db_path=None, api_url=None, transport=None):
    news_path = Path(news_path or os.getenv("DEMO_NEWS_PATH", ROOT.parent / "news.tsv"))
    db_path = Path(db_path or os.getenv("DEMO_DB_PATH", ROOT / ".data" / "sessions.sqlite3"))
    api_url = (api_url or os.getenv("RECOMMENDER_API_URL", "http://127.0.0.1:8000")).rstrip("/")
    parsed = urlsplit(api_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError("RECOMMENDER_API_URL must be an HTTP(S) API origin")

    @asynccontextmanager
    async def lifespan(app):
        app.state.catalog = load_catalog(news_path)
        app.state.store = SessionStore(db_path)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10, connect=3), transport=transport,
            follow_redirects=False, trust_env=False,
        ) as upstream:
            app.state.upstream = upstream
            yield

    app = FastAPI(title="Next Read · Session demo", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.middleware("http")
    async def same_origin_and_headers(request, call_next):
        origin = request.headers.get("origin")
        if request.method in {"POST", "DELETE"} and origin:
            if origin.rstrip("/") != str(request.base_url).rstrip("/"):
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "Cross-origin changes are not allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def get_session(request, response):
        token, created = app.state.store.session(request.cookies.get(COOKIE))
        if created:
            response.set_cookie(
                COOKIE, token, max_age=SESSION_SECONDS, httponly=True,
                samesite="lax", secure=os.getenv("DEMO_SECURE_COOKIE", "0") == "1",
            )
        return token

    async def state(token):
        history = app.state.store.history(token)
        catalog = app.state.catalog
        recommendations, error, latency = [], None, None
        try:
            result = await app.state.upstream.post(api_url + "/recommend", json={
                "user_history": history[-100:], "top_k": 4, "retrieve_k": 100,
            })
            result.raise_for_status()
            data = result.json()
            items = data["recommendations"]
            if not isinstance(items, list):
                raise ValueError("Invalid recommendation response")
            seen = set(history)
            for item in items:
                news_id = item["news_id"]
                if news_id in catalog and news_id not in seen:
                    recommendations.append(catalog[news_id])
                    seen.add(news_id)
                if len(recommendations) == 4:
                    break
            timing = data.get("latency_ms")
            if isinstance(timing, (int, float)) and math.isfinite(timing) and timing >= 0:
                latency = round(timing, 1)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            logger.warning("Recommendation API unavailable or returned an invalid response")
            error = ("Recommendations are paused. Start your recommender on port 8000, "
                     "or configure RECOMMENDER_API_URL for your AWS ALB. Your reading history is saved.")
        return {
            "session": token[:8], "read_count": len(history),
            "history": [catalog[n] for n in reversed(history) if n in catalog][:20],
            "recommendations": recommendations, "latency_ms": latency,
            "api_connected": error is None, "error": error,
        }

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/catalog")
    def catalog():
        # A balanced starter shelf drawn from the actual metadata, not new model predictions.
        categories = {}
        for article in app.state.catalog.values():
            bucket = categories.setdefault(article["category"], [])
            if len(bucket) < 4:
                bucket.append(article)
        articles = [article for bucket in categories.values() for article in bucket]
        return {"articles": articles, "categories": list(categories)}

    @app.get("/api/session")
    async def session(request: Request, response: Response):
        return await state(get_session(request, response))

    @app.post("/api/read")
    async def read(body: ReadRequest, request: Request, response: Response):
        if body.news_id not in app.state.catalog:
            raise HTTPException(404, "Article not in the demo catalog")
        token = get_session(request, response)
        app.state.store.read(token, body.news_id)
        return await state(token)

    @app.post("/api/session/reset")
    async def reset(request: Request, response: Response):
        token = get_session(request, response)
        app.state.store.clear(token)
        return await state(token)

    @app.post("/api/session/new")
    async def new_session(request: Request, response: Response):
        response.delete_cookie(COOKIE)
        token, _ = app.state.store.session(None)
        response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True,
                            samesite="lax", secure=os.getenv("DEMO_SECURE_COOKIE", "0") == "1")
        return await state(token)

    return app


app = create_app()
