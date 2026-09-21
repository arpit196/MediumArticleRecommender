# Two-Stage Personalized News Recommendation Engine

An end-to-end recommendation pipeline designed to surface relevant articles based on a user's historical reading behaviour and interests.

The system follows a **two-stage retrieval-and-ranking architecture**, separating the recommendation process into:

1. **Candidate Retrieval — high recall**
2. **Learning-to-Rank — high precision**

This architecture avoids scoring every document with an expensive ranking model while still producing highly personalized recommendations, making the system suitable for scaling to large document corpora.

---

## 🏗️ System Architecture

### 1. Candidate Retrieval

Documents are transformed into dense vector embeddings using a transformer-based embedding model. A user representation is generated from their historical reads and interests, with recent interactions contributing to the user profile.

The resulting user embedding is matched against document embeddings using vector similarity search.
The retrieval layer selects the **top-N candidate articles**, typically the top 100, and passes them to the ranking stage.
The system combines:

* **Dense semantic retrieval** using vector embeddings
* **Sparse lexical retrieval** using BM25
* **Reciprocal Rank Fusion (RRF)** to combine results from both retrieval strategies

This allows the retriever to capture both semantic relevance and exact keyword overlap.

### 2. Learning-to-Rank

Once candidate retrieval produces the most relevant articles, user-level and article-level features are generated for each candidate.
These features are passed to an **XGBoost Learning-to-Rank model**, which re-scores the candidates according to their predicted relevance to the user.
The highest-ranked articles are then returned as the final personalized recommendations.

For example:

```text
All Articles
     │
     ▼
Dense Retrieval ─────┐
                     ├──► Reciprocal Rank Fusion
BM25 Retrieval ──────┘              │
                                    ▼
                           Top-N Candidates
                                    │
                                    ▼
                         Feature Engineering
                                    │
                                    ▼
                        XGBoost LTR Ranker
                                    │
                                    ▼
                       Top-K Recommendations
```

---

## 🚀 Key Features & Engineering Highlights

### Hybrid Retrieval

Combines **semantic dense retrieval** using vector embeddings with **lexical sparse retrieval using BM25**.

Dense retrieval captures conceptual similarity between users and articles, while BM25 preserves exact keyword and terminology matching.

### Reciprocal Rank Fusion

Implements **Reciprocal Rank Fusion (RRF)** to combine candidates from the dense and sparse retrieval pipelines.

RRF merges rankings from heterogeneous retrieval systems (BM25 and vector search) without requiring their raw relevance scores to be directly comparable or normalized.

### Learning-to-Rank

Uses an **XGBoost ranking model** to re-rank retrieved candidates using user, document, similarity, behavioural, and positional features.

The ranking stage transforms a broad, high-recall candidate set into a smaller set of highly personalized recommendations.

### Personalized User Representations

User preferences are represented using historical article interactions and dense document embeddings, allowing recommendations to reflect both long-term interests and recent reading behaviour.

### Production-Style Inference API

The recommendation engine is exposed through **FastAPI**, separating offline model development from online inference.

The API loads the trained ranking model, vector index, metadata, and embedding artifacts at startup and serves recommendations through a low-latency HTTP endpoint.

### Advanced Evaluation

Recommendation quality is evaluated using ranking and retrieval metrics including:

* **NDCG@K — Normalized Discounted Cumulative Gain**
* **MRR — Mean Reciprocal Rank**
* **Recall@K**
* **Precision@K**

These metrics allow the retrieval and ranking stages to be evaluated independently rather than treating recommendation quality as a single black-box metric.


## Running the Recommendation Service

The inference service is implemented using **FastAPI**.

By default, the service runs on:

```text
http://localhost:8000
```

### Check Service Health

```bash
curl http://localhost:8000/health
```

The health endpoint verifies that the recommendation artifacts have been loaded successfully.

A typical response includes information about the loaded FAISS index, embeddings, news metadata, and XGBoost model.

### Check Service Readiness

```bash
curl http://localhost:8000/ready
```

Expected response:

```json
{
  "status": "ready"
}
```

---

## 📡 Request Recommendations

Send a user's reading history together with the required number of recommendations:

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_history": [
      "N55528",
      "N19639",
      "N61837",
      "N53526"
    ],
    "top_k": 5,
    "retrieve_k": 100
  }'
```

Where:

* `user_history` — previously viewed article IDs
* `retrieve_k` — number of candidates retrieved before re-ranking
* `top_k` — number of final recommendations returned

A typical response follows the structure:

```json
{
  "recommendations": [
    {
      "news_id": "N12345",
      "score": 0.87,
      "title": "Example Article",
      "category": "technology",
      "subcategory": "ai"
    }
  ],
  "latency_ms": 6.4,
  "top_k": 5,
  "retrieve_k": 100
}
```

---

# 🐳 Running with Docker

## 1. Build the Image

```bash
docker build -t rec-engine:latest .
```

## 2. Run the Container

```bash
docker run --rm \
  -p 8000:8000 \
  --name rec-engine \
  rec-engine:latest
```

Alternatively, if Docker Compose is configured:

```bash
docker compose up --build
```

## 3. Verify the Container

```bash
curl http://localhost:8000/health
```

Then request recommendations:

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_history": [
      "N55528",
      "N19639",
      "N61837"
    ],
    "top_k": 5,
    "retrieve_k": 100
  }'
```

> If Docker Compose maps container port `8000` to host port `8080`, replace `localhost:8000` with `localhost:8080`.

---

---

## 🎯 Design Goals

The project is designed around several practical recommendation-system principles:

* **High-recall retrieval** before expensive ranking
* Separation of **retrieval quality** from **ranking quality**
* Low-latency inference
* Reproducible offline evaluation
* Modular retrieval and ranking components
* Production-style API serving
* Containerized deployment

This separation makes it possible to independently improve the embedding model, retrieval strategy, ranking features, or LTR model without redesigning the entire recommendation pipeline.

---

## 📈 Future Improvements

Potential extensions include:

* Online A/B testing
* Real-time interaction features
* Feature-store integration
* Model and embedding versioning
* Prometheus/Grafana monitoring
* Automated retraining pipelines
* Kubernetes deployment
* Retrieval and ranking drift detection
* Recommendation diversity and novelty constraints
* Cold-start strategies for new users and new articles
