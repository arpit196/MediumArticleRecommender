# app.py
from __future__ import annotations
from datetime import datetime
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import csv
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("recommender-api")

LATENCY_LOG = Path("latency_log.csv")

if not LATENCY_LOG.exists():
    with open(LATENCY_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "latency_ms",
            "history_length",
            "retrieve_k",
            "top_k",
            "recommendations_returned"
        ])


# =========================
# Config
# =========================
@dataclass(frozen=True)
class Settings:
    NEWS_METADATA_PATH: str = os.getenv("NEWS_METADATA_PATH", "news.tsv")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "artifacts/recommender-v1.0/faiss_hnsw_index.bin")
    EMBEDDINGS_MAP_PATH: str = os.getenv("EMBEDDINGS_MAP_PATH", "artifacts/recommender-v1.0/embeddings_map.pkl")
    XGB_MODEL_PATH: str = os.getenv("XGB_MODEL_PATH", "artifacts/recommender-v1.0/xgb_news_rerankerx.json")
    CTR_MAP_PATH: str = os.getenv("CTR_MAP_PATH", "artifacts/recommender-v1.0/impression_ctr_map.json")  # optional
    TOP_K_DEFAULT: int = int(os.getenv("TOP_K_DEFAULT", "10"))
    RETRIEVE_K_DEFAULT: int = int(os.getenv("RETRIEVE_K_DEFAULT", "100"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


SETTINGS = Settings()


# =========================
# Schemas
# =========================
class RecommendRequest(BaseModel):
    user_history: List[str] = Field(
        ...,
        description="List of previously read news IDs in chronological order (oldest -> newest).",
        min_items=0,
    )
    top_k: int = Field(default=10, ge=1, le=100)
    retrieve_k: int = Field(default=100, ge=1, le=500)
    impression_id: Optional[str] = Field(
        default=None,
        description="Optional impression/session id. Used only if a CTR lookup table exists.",
    )


class RecommendationItem(BaseModel):
    news_id: str
    score: float
    title: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None


class RecommendResponse(BaseModel):
    recommendations: List[RecommendationItem]
    latency_ms: float
    top_k: int
    retrieve_k: int


# =========================
# Data / model utilities
# =========================
class NewsStore:
    """
    Loads news metadata needed at inference time.
    Required columns:
      - news_id
      - category
      - subcategory
      - title
    """

    def __init__(self, news_path: str):
        self.news_path = news_path
        self.df_news = self._load_news()
        self.df_news["news_id"] = self.df_news["news_id"].astype(str)

        required = {"news_id", "category", "subcategory", "title"}
        missing = required - set(self.df_news.columns)
        if missing:
            raise ValueError(
                f"News metadata missing required columns: {sorted(missing)}"
            )

        self.news_cat_map = dict(
            zip(self.df_news["news_id"], self.df_news["category"].fillna(""))
        )
        self.news_subcat_map = dict(
            zip(self.df_news["news_id"], self.df_news["subcategory"].fillna(""))
        )
        self.news_title_map = dict(
            zip(self.df_news["news_id"], self.df_news["title"].fillna(""))
        )

    def _load_news(self) -> pd.DataFrame:
        news_cols = [
            "news_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ]
        if not os.path.exists(self.news_path):
            raise FileNotFoundError(
                f"News metadata file not found: {self.news_path}"
            )

        if self.news_path.endswith(".parquet"):
            return pd.read_parquet(self.news_path)
        if self.news_path.endswith(".csv") or self.news_path.endswith(".tsv"):
            sep = "\t" if self.news_path.endswith(".tsv") else ","
            return pd.read_csv(self.news_path, sep=sep,header=None,names=news_cols,usecols=["news_id", "category", "subcategory", "title", "abstract"],)

        raise ValueError(
            "Unsupported news file format. Use .parquet, .csv, or .tsv."
        )

    def get_news_row(self, news_id: str) -> Dict[str, Any]:
        return {
            "news_id": news_id,
            "title": self.news_title_map.get(news_id),
            "category": self.news_cat_map.get(news_id),
            "subcategory": self.news_subcat_map.get(news_id),
        }


class HybridRetriever:
    """
    Serving-time retriever using:
      - FAISS index
      - saved article embedding map

    This matches the notebook structure:
      - build_article_index(...) during offline training
      - faiss.write_index(...)
      - pickle.dump(embeddings_map,...)
    """

    def __init__(self, index: faiss.Index, embeddings_map: Dict[str, np.ndarray], news_ids: List[str]):
        self.index = index
        self.embeddings_map = embeddings_map
        self.news_ids = news_ids

    @staticmethod
    def load(index_path: str, embeddings_map_path: str) -> "HybridRetriever":
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(embeddings_map_path):
            raise FileNotFoundError(f"Embeddings map not found: {embeddings_map_path}")

        index = faiss.read_index(index_path)

        with open(embeddings_map_path, "rb") as f:
            embeddings_map = pickle.load(f)

        if not isinstance(embeddings_map, dict):
            raise ValueError("embeddings_map.pkl must contain a dict of {news_id: vector}")

        news_ids = list(embeddings_map.keys())
        logger.info("Loaded FAISS index with %s vectors.", index.ntotal)
        logger.info("Loaded embeddings map with %s items.", len(embeddings_map))
        return HybridRetriever(index=index, embeddings_map=embeddings_map, news_ids=news_ids)

    def generate_user_vector(self, history_news_ids: List[str]) -> np.ndarray:
        """
        Same idea as the notebook:
        - look up history embeddings
        - apply recency weighting
        - return normalized user vector
        """
        valid_vectors = [
            self.embeddings_map[nid]
            for nid in history_news_ids
            if nid in self.embeddings_map
        ]

        if not valid_vectors:
            # 384 is the dimension used in the notebook fallback.
            # If your embedding dim differs, change this fallback to the actual size.
            dim = len(next(iter(self.embeddings_map.values())))
            return np.zeros(dim, dtype=np.float32)

        hist_matrix = np.array(valid_vectors, dtype=np.float32)
        weights = np.exp(np.linspace(-1.0, 0.0, len(valid_vectors))).astype(np.float32)
        weights /= np.sum(weights)

        user_vector = np.sum(hist_matrix * weights[:, None], axis=0).astype(np.float32)
        norm = np.linalg.norm(user_vector)
        return (user_vector / (norm + 1e-9)).astype(np.float32)

    def retrieve_candidates(self, user_vector: np.ndarray, top_k: int = 100) -> List[str]:
        if np.all(user_vector == 0):
            return self.news_ids[:top_k]

        query = np.array([user_vector], dtype=np.float32)
        _, indices = self.index.search(query, top_k)
        return [self.news_ids[i] for i in indices[0] if 0 <= i < len(self.news_ids)]


class RecommendationEngine:
    def __init__(
        self,
        retriever: HybridRetriever,
        news_store: NewsStore,
        xgb_model: xgb.Booster,
        ctr_map: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.retriever = retriever
        self.news_store = news_store
        self.model = xgb_model
        self.ctr_map = ctr_map or {}

        # Pre-normalize embeddings for fast similarity features.
        self.normalized_embeddings: Dict[str, np.ndarray] = {}
        for news_id, vec in self.retriever.embeddings_map.items():
            v = np.asarray(vec, dtype=np.float32)
            n = np.linalg.norm(v)
            self.normalized_embeddings[news_id] = (v / (n + 1e-9)).astype(np.float32) if n > 0 else v

    @staticmethod
    def _safe_entropy(probabilities: List[float]) -> float:
        if not probabilities:
            return 0.0
        p = np.asarray(probabilities, dtype=np.float32)
        return float(-np.sum(p * np.log2(p + 1e-9)))

    def _get_user_profile(self, history_ids: List[str]) -> Tuple:
        """
        Mirrors the notebook's feature logic.
        Returns:
          hist_len, cat_affinity_dict, top_user_cat, subcat_affinity_dict,
          top_user_subcat, user_entropy, user_vec, recency_user_vec, hist_matrix
        """
        news_cat_map = self.news_store.news_cat_map
        news_subcat_map = self.news_store.news_subcat_map

        hist_len = len(history_ids)

        cat_counts: Dict[str, int] = {}
        subcat_counts: Dict[str, int] = {}

        for hid in history_ids:
            if hid in news_cat_map:
                cat_counts[news_cat_map[hid]] = cat_counts.get(news_cat_map[hid], 0) + 1
            if hid in news_subcat_map:
                subcat_counts[news_subcat_map[hid]] = subcat_counts.get(news_subcat_map[hid], 0) + 1

        total_cats = sum(cat_counts.values())
        total_subcats = sum(subcat_counts.values())

        if total_cats > 0:
            cat_affinity_dict = {c: cnt / total_cats for c, cnt in cat_counts.items()}
            top_user_cat = max(cat_counts.items(), key=lambda x: x[1])[0]
            user_entropy = self._safe_entropy(list(cat_affinity_dict.values()))
        else:
            cat_affinity_dict, top_user_cat, user_entropy = {}, "", 0.0

        if total_subcats > 0:
            subcat_affinity_dict = {sc: cnt / total_subcats for sc, cnt in subcat_counts.items()}
            top_user_subcat = max(subcat_counts.items(), key=lambda x: x[1])[0]
        else:
            subcat_affinity_dict, top_user_subcat = {}, ""

        hist_vectors = [
            self.normalized_embeddings[hid]
            for hid in history_ids
            if hid in self.normalized_embeddings
        ]

        if hist_vectors:
            hist_matrix = np.array(hist_vectors, dtype=np.float32)

            user_vec = np.mean(hist_matrix, axis=0)
            user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-9)

            weights = np.exp(np.linspace(-1.0, 0.0, len(hist_vectors))).astype(np.float32)
            weights /= np.sum(weights)
            recency_user_vec = np.sum(hist_matrix * weights[:, None], axis=0).astype(np.float32)
            recency_user_vec = recency_user_vec / (np.linalg.norm(recency_user_vec) + 1e-9)
        else:
            dim = len(next(iter(self.normalized_embeddings.values())))
            hist_matrix = None
            user_vec = np.zeros(dim, dtype=np.float32)
            recency_user_vec = np.zeros(dim, dtype=np.float32)

        return (
            hist_len,
            cat_affinity_dict,
            top_user_cat,
            subcat_affinity_dict,
            top_user_subcat,
            user_entropy,
            user_vec,
            recency_user_vec,
            hist_matrix,
        )

    def _feature_row(
        self,
        candidate_id: str,
        pos: int,
        total_candidates: int,
        profile: Tuple,
        impression_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        (
            hist_len,
            cat_affinity_dict,
            top_user_cat,
            subcat_affinity_dict,
            top_user_subcat,
            user_entropy,
            user_vec,
            recency_user_vec,
            hist_matrix,
        ) = profile

        cand_cat = self.news_store.news_cat_map.get(candidate_id, "")
        cand_subcat = self.news_store.news_subcat_map.get(candidate_id, "")
        c_vec = self.normalized_embeddings.get(candidate_id, None)

        if c_vec is not None:
            dense_sim = float(np.dot(c_vec, user_vec))
            recency_sim = float(np.dot(c_vec, recency_user_vec))
            if hist_matrix is not None:
                sims = np.dot(hist_matrix, c_vec)
                max_sim = float(np.max(sims)) if len(sims) else 0.0
                top3_sim = float(np.mean(np.sort(sims)[-3:])) if len(sims) >= 3 else float(np.mean(sims))
            else:
                max_sim = 0.0
                top3_sim = 0.0
        else:
            dense_sim = 0.0
            recency_sim = 0.0
            max_sim = 0.0
            top3_sim = 0.0

        cat_affinity = cat_affinity_dict.get(cand_cat, 0.0)
        subcat_affinity = subcat_affinity_dict.get(cand_subcat, 0.0)

        # In the notebook, global_ctr came from an impression->candidate CTR map.
        # At serving time, we use it if available; otherwise we default to 0.0.
        global_ctr = 0.0
        if impression_id and impression_id in self.ctr_map:
            global_ctr = float(self.ctr_map[impression_id].get(candidate_id, 0.0))

        return {
            "candidate_id": candidate_id,
            "dense_cosine_sim": np.float32(dense_sim),
            "max_cosine_sim": np.float32(max_sim),
            "recency_weighted_sim": np.float32(recency_sim),
            "category_match": np.float32(1.0 if cand_cat == top_user_cat else 0.0),
            "user_cat_affinity": np.float32(cat_affinity),
            "subcategory_match": np.float32(1.0 if cand_subcat == top_user_subcat else 0.0),
            "top3_sim": np.float32(top3_sim),
            "user_subcat_affinity": np.float32(subcat_affinity),
            "is_new_category": np.float32(1.0 if cand_cat not in cat_affinity_dict else 0.0),
            "user_hist_len": np.int16(hist_len),
            "user_history_entropy": np.float32(user_entropy),
            "impression_position": np.int16(pos),
            "normalized_imp_position": np.float32(pos / total_candidates if total_candidates > 0 else 0.0),
            "global_ctr": np.float32(global_ctr),
        }

    def recommend(
        self,
        user_history: List[str],
        top_k: int = 10,
        retrieve_k: int = 100,
        impression_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []

        profile = self._get_user_profile(user_history)
        user_vec = profile[6]

        candidate_ids = self.retriever.retrieve_candidates(user_vec, top_k=retrieve_k)

        if not candidate_ids:
            return []

        rows = []
        for pos, cid in enumerate(candidate_ids):
            # Filter out items already seen by the user
            if cid in set(user_history):
                continue
            rows.append(self._feature_row(cid, pos, len(candidate_ids), profile, impression_id))

        if not rows:
            # fallback: return raw candidates excluding history
            out = []
            for cid in candidate_ids:
                if cid in set(user_history):
                    continue
                meta = self.news_store.get_news_row(cid)
                out.append(
                    {
                        "news_id": cid,
                        "score": 0.0,
                        "title": meta["title"],
                        "category": meta["category"],
                        "subcategory": meta["subcategory"],
                    }
                )
                if len(out) >= top_k:
                    break
            return out

        df = pd.DataFrame(rows)

        feature_cols = [
            "dense_cosine_sim",
            "max_cosine_sim",
            "recency_weighted_sim",
            "category_match",
            "user_cat_affinity",
            "subcategory_match",
            "top3_sim",
            "user_subcat_affinity",
            "is_new_category",
            "user_hist_len",
            "user_history_entropy",
            "impression_position",
            "normalized_imp_position",
            "global_ctr",
        ]

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing inference features: {missing}")

        X = df[feature_cols].to_numpy(dtype=np.float32)
        dmat = xgb.DMatrix(X, feature_names=feature_cols)
        scores = self.model.predict(dmat)

        df["score"] = scores
        df.sort_values("score", ascending=False, inplace=True)

        recommendations = []
        for _, row in df.head(top_k).iterrows():
            meta = self.news_store.get_news_row(str(row["candidate_id"]))
            recommendations.append(
                {
                    "news_id": str(row["candidate_id"]),
                    "score": float(row["score"]),
                    "title": meta["title"],
                    "category": meta["category"],
                    "subcategory": meta["subcategory"],
                }
            )
        return recommendations

    def health_payload(self) -> Dict[str, Any]:
        return {
            "faiss_vectors": int(self.retriever.index.ntotal),
            "num_embeddings": int(len(self.retriever.embeddings_map)),
            "num_news": int(len(self.news_store.df_news)),
            "xgb_model_loaded": True,
        }


# =========================
# Optional CTR map loader
# =========================
def load_ctr_map(path: str) -> Dict[str, Dict[str, float]]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning("Could not load CTR map from %s: %s", path, e)
    return {}


# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="Recommendation Service",
    version="1.0.0",
    description="FastAPI service for hybrid news recommendation + XGBoost ranking.",
)

ENGINE: Optional[RecommendationEngine] = None
STARTUP_TIME: Optional[float] = None


@app.on_event("startup")
def startup_event() -> None:
    global ENGINE, STARTUP_TIME
    STARTUP_TIME = time.time()

    logger.info("Loading artifacts...")
    news_store = NewsStore(SETTINGS.NEWS_METADATA_PATH)
    retriever = HybridRetriever.load(
        SETTINGS.FAISS_INDEX_PATH, SETTINGS.EMBEDDINGS_MAP_PATH
    )

    if not os.path.exists(SETTINGS.XGB_MODEL_PATH):
        raise FileNotFoundError(f"XGBoost model not found: {SETTINGS.XGB_MODEL_PATH}")

    model = xgb.Booster()
    model.load_model(SETTINGS.XGB_MODEL_PATH)

    ctr_map = load_ctr_map(SETTINGS.CTR_MAP_PATH)

    ENGINE = RecommendationEngine(
        retriever=retriever,
        news_store=news_store,
        xgb_model=model,
        ctr_map=ctr_map,
    )

    logger.info("Startup complete in %.2fs", time.time() - STARTUP_TIME)


@app.get("/health")
def health() -> Dict[str, Any]:
    if ENGINE is None:
        return {"status": "not_ready"}
    return {"status": "ok", **ENGINE.health_payload()}


@app.get("/ready")
def ready() -> Dict[str, Any]:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()

    if req.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0")

    try:
        recs = ENGINE.recommend(
            user_history=req.user_history,
            top_k=req.top_k,
            retrieve_k=req.retrieve_k,
            impression_id=req.impression_id,
        )
    except Exception as e:
        logger.exception("Recommendation failed")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - start) * 1000.0
    with open(LATENCY_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            round(latency_ms, 2),
            len(req.user_history),
            req.retrieve_k,
            req.top_k,
            len(recs)
        ])

    return RecommendResponse(
        recommendations=[RecommendationItem(**r) for r in recs],
        latency_ms=latency_ms,
        top_k=req.top_k,
        retrieve_k=req.retrieve_k,
    )


@app.post("/warmup")
def warmup() -> Dict[str, Any]:
    """
    Optional endpoint to warm the service and measure a sample inference.
    """
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sample_history = list(ENGINE.news_store.df_news["news_id"].head(5).astype(str))
    t0 = time.perf_counter()
    _ = ENGINE.recommend(sample_history, top_k=5, retrieve_k=25)
    ms = (time.perf_counter() - t0) * 1000.0
    return {"status": "warmed_up", "sample_latency_ms": ms}


if __name__ == "__main__":
    uvicorn.run("app:app", host=SETTINGS.HOST, port=SETTINGS.PORT, reload=False)