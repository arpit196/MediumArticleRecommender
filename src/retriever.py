import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class HybridRetriever:

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.news_ids = []
        self.encoder.to("cpu")

    def build_article_index(self, df_news: pd.DataFrame):
        """Creates dense vectors for all news articles and builds HNSW FAISS index."""
        self.news_ids = df_news["news_id"].tolist()

        # Combine title, subcategory, and abstract
        texts = (
            df_news["category"].fillna("")
            + " "
            + df_news["subcategory"].fillna("")
            + ": "
            + df_news["title"].fillna("")
        ).tolist()

        print("Encoding articles into dense vectors...")
        embeddings = self.encoder.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        )
        embeddings_map = {df_news["news_id"].iloc[idx]:embeddings[idx] for idx in range(len(df_news))}
        self.embeddings_map = embeddings_map

        # Build HNSW Index (High speed, high recall ANN)
        d = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(d, 32)
        self.index.add(np.array(embeddings).astype("float32"))
        print(f"FAISS Index built with {self.index.ntotal} articles.")

    def generate_user_vector(
        self, history_news_ids: list, news_embedding_map: dict
    ) -> np.ndarray:
        """Computes a time-decay weighted user representation from historical read IDs.

        Ensures NO target leakage by only accepting past history IDs.
        """
        valid_vectors = [
            news_embedding_map[nid]
            for nid in history_news_ids
            if nid in news_embedding_map
        ]

        if not valid_vectors:
            # Fallback for cold-start users (return zero vector or global average)
            return np.zeros(384, dtype="float32")

        # Apply exponential decay (newer history items get higher weight)
        weights = np.exp(np.linspace(-1.0, 0.0, len(valid_vectors)))
        weights /= np.sum(weights)

        user_vector = np.sum(
            [w * vec for w, vec in zip(weights, valid_vectors)], axis=0
        )
        # Re-normalize vector
        return user_vector / (np.linalg.norm(user_vector) + 1e-9)

    def retrieve_candidates(self, user_vector: np.ndarray, top_k=100) -> list:
        """Fetches top-K candidate news IDs for a given user representation."""
        if np.all(user_vector == 0):
            # Cold start fallback
            return self.news_ids[:top_k]

        query = np.array([user_vector]).astype("float32")
        distances, indices = self.index.search(query, top_k)

        retrieved_ids = [self.news_ids[idx] for idx in indices[0]]
        return retrieved_ids