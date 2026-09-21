FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEWS_METADATA_PATH=/app/news.tsv \
    FAISS_INDEX_PATH=/app/faiss_hnsw_index.bin \
    EMBEDDINGS_MAP_PATH=/app/embeddings_map.pkl \
    XGB_MODEL_PATH=/app/xgb_news_rerankerx.json \
    CTR_MAP_PATH=/app/impression_ctr_map.json

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application
COPY app.py .

# Copy inference artifacts
COPY news.tsv .
COPY faiss_hnsw_index.bin .
COPY embeddings_map.pkl .
COPY xgb_news_rerankerx.json .

# Optional CTR file - only include this COPY if the file actually exists
# COPY impression_ctr_map.json .

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8000/ready || exit 1

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
