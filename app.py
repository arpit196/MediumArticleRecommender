import os
import faiss
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import json

app = Flask(__name__)

# Global runtime variables for holding retrieval state
MODEL = None
FAISS_INDEX = None
BM25_ENGINE = None
ARTICLES_METADATA = None

def load_retrieval_assets():
    """
    Loads all heavyweight indexes into memory once upon worker instantiation.
    """
    global MODEL, FAISS_INDEX, BM25_ENGINE, ARTICLES_METADATA
    print("Initializing engine states and reading binary indexes...")
    
    # 1. Load Sentence Transformer local path
    model_path = os.getenv("MODEL_PATH", "./assets/all_mini_lm")
    MODEL = SentenceTransformer(model_path, local_files_only=True)
    MODEL.to('cpu') # Enforce strict thread-safe CPU memory allocation
    
    # 2. Load FAISS Quantized Vector Space
    faiss_path = os.getenv("FAISS_PATH", "./assets/my_index_quantized.faiss")
    FAISS_INDEX = faiss.read_index(faiss_path)
    FAISS_INDEX.nprobe = 10
    
    # 3. Load Article Meta Mapper for instant O(1) attribute responses
    meta_path = os.getenv("METADATA_PATH", "./assets/articles_meta.json")
    with open(meta_path, 'r') as f:
        # Expected shape: {"0": {"title": "X", "url": "Y"}, "1": {...}}
        ARTICLES_METADATA = json.load(f)
        
    # 4. Initialize BM25 engine using tokenized strings
    tokenized_corpus_path = os.getenv("TOKENIZED_CORPUS_PATH", "./assets/tokenized_corpus.json")
    with open(tokenized_corpus_path, 'r') as f:
        tokenized_corpus = json.load(f)
    BM25_ENGINE = BM25Okapi(tokenized_corpus)
    print("All search engines successfully initialized.")

# --- CRITICAL FIX: Trigger load immediately so assets populate when Gunicorn imports the app ---
load_retrieval_assets()

def reciprocal_rank_fusion(dense_indices, sparse_indices, k=60, top_n=50):
    """
    Computes standard RRF across dense semantic and sparse lexical indices.
    """
    rrf_scores = {}
    
    # Process Dense Ranks
    for rank, idx in enumerate(dense_indices):
        idx = int(idx)
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))
        
    # Process Sparse Ranks
    for rank, idx in enumerate(sparse_indices):
        idx = int(idx)
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))
        
    # Sort candidates by combined score descending
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates[:top_n]

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/api/retrieve', methods=['POST'])
def retrieve_candidates():
    data = request.get_json() or {}
    query_text = data.get("query", "").strip()
    
    if not query_text:
        return jsonify({"error": "Empty search parameters provided"}), 400
        
    try:
        # Step A: Generate dense embedding vector and lookup FAISS index
        query_emb = MODEL.encode([query_text], convert_to_numpy=True).astype('float32')
        _, dense_indices = FAISS_INDEX.search(query_emb, k=100)
        dense_results = dense_indices[0].tolist()
        
        # Step B: Parse string tokens and get top structural matches from BM25
        tokenized_query = query_text.lower().split(" ")
        bm25_scores = BM25_ENGINE.get_scores(tokenized_query)
        sparse_results = np.argpartition(bm25_scores, -100)[-100:]
        sparse_results = sparse_results[np.argsort(bm25_scores[sparse_results])][::-1].tolist()
        
        # Step C: Intersect and sort results via Reciprocal Rank Fusion (RRF)
        fused_candidates = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=20)
        
        # Step D: Enrich data frame mapping responses out to clients
        response_payload = []
        for index_id, rrf_score in fused_candidates:
            meta = ARTICLES_METADATA.get(str(index_id), {"title": "Unknown Document", "url": "#"})
            response_payload.append({
                "article_id": index_id,
                "rrf_score": round(rrf_score, 5),
                "title": meta.get("title"),
                "url": meta.get("url")
            })
            
        return jsonify({"results": response_payload}), 200
        
    except Exception as e:
        return jsonify({"error": f"An engine fault occurred during retrieval: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)