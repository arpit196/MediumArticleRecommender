import numpy as np
import pandas as pd
from collections import defaultdict
import numpy as np
import pandas as pd
import xgboost as xgb
import math
from collections import Counter, defaultdict
import glob

def calculate_metrics(recommended_ids, ground_truth_ids, k_list=[5, 10, 20]):
    """Calculates comprehensive evaluation metrics for a single query.

    Parameters:
    - recommended_ids: Ordered list of document IDs returned by the system.
    - ground_truth_ids: Set or list of true positive document IDs for this query.
    - k_list: List of cut-offs (@K) to evaluate.
    """
    metrics = {}
    ground_truth_set = set(ground_truth_ids)

    if not ground_truth_set:
        return None  # Skip if no ground truth exists for this query

    # Pre-calculate relevance binary array for NDCG/MRR up to max K
    max_k = max(k_list)
    relevance_binary = [
        1 if doc_id in ground_truth_set else 0
        for doc_id in recommended_ids[:max_k]
    ]

    for k in k_list:
        # Cut recommendations and relevance to current K
        rec_at_k = recommended_ids[:k]
        rel_at_k = relevance_binary[:k]

        # 1. Precision@K = (Relevant Recs) / (Total Recs at K)
        num_relevant = sum(rel_at_k)
        precision = num_relevant / k

        # 2. Recall@K = (Relevant Recs) / (Total True Positives)
        recall = num_relevant / len(ground_truth_set)

        # 3. F1@K = Harmonic Mean of Precision and Recall
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        # 4. NDCG@K (Normalized Discounted Cumulative Gain)
        # Accounts for the exact position of the relevant documents
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(rel_at_k)])
        idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(k, len(ground_truth_set)))])
        ndcg = dcg / idcg if idcg > 0 else 0.0

        metrics[f"Precision@{k}"] = precision
        metrics[f"Recall@{k}"] = recall
        metrics[f"F1@{k}"] = f1
        metrics[f"NDCG@{k}"] = ndcg

    # 5. MRR (Mean Reciprocal Rank) - Evaluates where the FIRST relevant item shows up
    try:
        first_relevant_rank = relevance_binary.index(1) + 1
        mrr = 1.0 / first_relevant_rank
    except ValueError:
        mrr = 0.0  # No relevant item found in the top max_k results

    metrics["MRR"] = mrr

    return metrics




def evaluate_xgb_ranker_pipeline(
    df_behaviors,
    df_news,
    retriever,
    model,
    feature_cols,
    ground_truth_dict,
    article_ctr_map=None,
    k_list=[5, 10],
):
    """Evaluates pure XGBRanker model performance on impression sessions without RRF or ensemble mixing."""
    # 1. Pre-normalize all article embeddings once
    normalized_embeddings = {}
    for news_id, vec in retriever.embeddings_map.items():
        norm = np.linalg.norm(vec)
        normalized_embeddings[news_id] = (
            vec / (norm + 1e-9) if norm > 0 else vec
        )

    # 2. Pre-build Category/Subcategory lookup
    news_cat_map = {}
    for r in df_news.itertuples():
        cat = getattr(r, "category", "") or ""
        subcat = getattr(r, "subcategory", "") or ""
        news_cat_map[r.news_id] = (cat, subcat)

    user_cat_profile_cache = {}
    all_session_metrics = []

    # 3. Iterate through behaviors
    for row in df_behaviors.itertuples(index=False):
        imp_id = str(
            getattr(row, "impression_id", getattr(row, "Impression_ID", row[0]))
        )

        if imp_id not in ground_truth_dict or not ground_truth_dict[imp_id]:
            continue

        history_ids = (
            row.history.split(" ") if isinstance(row.history, str) else []
        )
        impression_items = row.impressions.split(" ")
        candidate_ids = [item.rsplit("-", 1)[0] for item in impression_items]
        ground_truth_ids = ground_truth_dict[imp_id]

        valid_candidate_ids = [
            cid for cid in candidate_ids if cid in news_cat_map
        ]
        if not valid_candidate_ids:
            continue

        # --- User Profile Computations ---
        hist_key = tuple(history_ids)
        if hist_key not in user_cat_profile_cache:
            cat_counts = Counter()
            for hid in history_ids:
                if hid in news_cat_map:
                    cat, _ = news_cat_map[hid]
                    if cat:
                        cat_counts[cat] += 1
            user_cat_profile_cache[hist_key] = cat_counts
        else:
            cat_counts = user_cat_profile_cache[hist_key]

        user_vector = retriever.generate_user_vector(
            history_ids, retriever.embeddings_map
        )
        user_norm = np.linalg.norm(user_vector)
        if user_norm > 0:
            user_vector = user_vector / user_norm

        # --- Extract Candidate Features for current session ---
        feature_rows = []
        for pos, cid in enumerate(valid_candidate_ids):
            cat, subcat = news_cat_map[cid]

            # Dense similarity
            vec = normalized_embeddings.get(cid, None)
            dense_sim = float(np.dot(user_vector, vec)) if vec is not None else 0.0

            # Category features
            cat_affinity = float(cat_counts.get(cat, 0))
            user_hist_len = float(len(history_ids))
            global_ctr = (
                float(article_ctr_map.get(cid, 0.0))
                if article_ctr_map
                else 0.0
            )

            # Build feature dictionary aligned with model columns
            feat_dict = {
                "dense_cosine_sim": dense_sim,
                "user_cat_affinity": cat_affinity,
                "user_hist_len": user_hist_len,
                "impression_position": float(pos)
            }
            feature_rows.append([feat_dict.get(col, 0.0) for col in feature_cols])

        # Convert to matrix and score with XGBRanker
        X_session = np.array(feature_rows, dtype=np.float32)

        # Handle both XGBoost Booster API and Scikit-Learn API
        if isinstance(model, xgb.Booster):
            dtest = xgb.DMatrix(X_session, feature_names=feature_cols)
            scores = model.predict(dtest)
        else:
            scores = model.predict(X_session)

        # Sort candidate IDs by XGBRanker predicted score descending
        sorted_indices = np.argsort(scores)[::-1]
        final_ranked_ids = [valid_candidate_ids[i] for i in sorted_indices]

        # Compute ranking metrics
        session_metric = calculate_metrics(
            final_ranked_ids, ground_truth_ids, k_list=k_list
        )
        if session_metric:
            all_session_metrics.append(session_metric)

    # 4. Aggregate metrics across all sessions
    if not all_session_metrics:
        return {}

    aggregated_metrics = {}
    metric_keys = all_session_metrics[0].keys()
    for key in metric_keys:
        aggregated_metrics[key] = float(
            np.mean([m[key] for m in all_session_metrics])
        )

    return aggregated_metrics



# --- 2B. Impression-Filtered Evaluation Function ---
def evaluate_xgb_ranker_from_parquet(
    chunk_files,
    model,
    feature_cols,
    target_imp_set,
    candidate_col="candidate_id",
    k_list=[10, 20],
):
  """Evaluates model strictly on the target impression IDs across parquet chunks."""
  required_cols = list(
      set(feature_cols + ["impression_id", candidate_col, "label"])
  )

  all_session_metrics = []

  for file_path in chunk_files:
    df_chunk = pd.read_parquet(file_path, columns=required_cols)

    # Filter chunk to validation impression_ids only
    df_chunk = df_chunk[df_chunk["impression_id"].astype(np.int32).isin(target_imp_set)]

    if df_chunk.empty:
      continue

    df_chunk.sort_values("impression_id", inplace=True)
    df_chunk.reset_index(drop=True, inplace=True)

    X_val = df_chunk[feature_cols].to_numpy(dtype=np.float32)

    if isinstance(model, xgb.Booster):
      dmatrix_chunk = xgb.DMatrix(X_val, feature_names=feature_cols)
      df_chunk["score"] = model.predict(dmatrix_chunk)
      del dmatrix_chunk
    else:
      df_chunk["score"] = model.predict(X_val)

    del X_val

    # NumPy slicing for session metrics
    imp_ids = df_chunk["impression_id"].to_numpy()
    candidate_ids = df_chunk[candidate_col].to_numpy()
    labels = df_chunk["label"].to_numpy()
    scores = df_chunk["score"].to_numpy()

    _, start_indices, counts = np.unique(
        imp_ids, return_index=True, return_counts=True
    )

    for start_idx, count in zip(start_indices, counts):
      end_idx = start_idx + count

      sess_scores = scores[start_idx:end_idx]
      sess_labels = labels[start_idx:end_idx]
      sess_cands = candidate_ids[start_idx:end_idx]

      if not (sess_labels.any() and not sess_labels.all()):
        continue

      order = np.argsort(-sess_scores)
      ranked_cands = sess_cands[order].tolist()
      ground_truth_ids = set(sess_cands[sess_labels == 1].tolist())

      session_metric = calculate_metrics(
          ranked_cands, ground_truth_ids, k_list=k_list
      )
      if session_metric:
        all_session_metrics.append(session_metric)

    del df_chunk
    gc.collect()

  if not all_session_metrics:
    return {}

  aggregated_metrics = {}
  for key in all_session_metrics[0].keys():
    aggregated_metrics[key] = float(
        np.mean([m[key] for m in all_session_metrics])
    )

  return aggregated_metrics


from collections import Counter, defaultdict
def evaluate_impression_pipeline(
    df_behaviors,
    df_news,
    retriever,
    ground_truth_dict,
    k_list=[10, 20],
    rrf_k=60,
    cat_weight=1.0,
    subcat_weight=2.0,
):
    """Evaluates the hybrid (Dense Vector + Category Affinity) candidate re-ranker across impression sessions."""
    # 1. Pre-normalize all article embeddings once
    normalized_embeddings = {}
    for news_id, vec in retriever.embeddings_map.items():
        norm = np.linalg.norm(vec)
        normalized_embeddings[news_id] = (
            vec / (norm + 1e-9) if norm > 0 else vec
        )

    # 2. Pre-build O(1) Category & Subcategory lookup map
    # Maps: news_id -> (category, subcategory)
    news_cat_map = {}
    for r in df_news.itertuples():
        cat = getattr(r, "category", "") or ""
        subcat = getattr(r, "subcategory", "") or ""
        news_cat_map[r.news_id] = (cat, subcat)

    # Pre-cache user category affinity profiles to avoid redundant counter loops
    user_cat_profile_cache = {}

    all_session_metrics = []

    # 3. Iterate through behaviors
    for row in df_behaviors.itertuples(index=False):
        imp_id = str(
            getattr(row, "impression_id", getattr(row, "Impression_ID", row[0]))
        )

        if imp_id not in ground_truth_dict or not ground_truth_dict[imp_id]:
            continue

        history_ids = (
            row.history.split(" ") if isinstance(row.history, str) else []
        )
        impression_items = row.impressions.split(" ")

        candidate_ids = [item.rsplit("-", 1)[0] for item in impression_items]
        ground_truth_ids = ground_truth_dict[imp_id]

        # Filter valid candidates present in dataset metadata
        valid_candidate_ids = [
            cid for cid in candidate_ids if cid in news_cat_map
        ]
        if not valid_candidate_ids:
            continue

        # --- A. CATEGORY / SUBCATEGORY AFFINITY RETRIEVAL ---
        hist_key = tuple(history_ids)

        if hist_key not in user_cat_profile_cache:
            cat_counts = Counter()
            subcat_counts = Counter()

            for hid in history_ids:
                if hid in news_cat_map:
                    cat, subcat = news_cat_map[hid]
                    if cat:
                        cat_counts[cat] += 1
                    if subcat:
                        subcat_counts[subcat] += 1

            user_cat_profile_cache[hist_key] = (cat_counts, subcat_counts)
        else:
            cat_counts, subcat_counts = user_cat_profile_cache[hist_key]

        cat_ranked_ids = []
        if cat_counts or subcat_counts:
            candidate_scores = []

            for cid in valid_candidate_ids:
                cat, subcat = news_cat_map[cid]

                # Calculate weighted affinity score
                score = (cat_counts.get(cat, 0) * cat_weight) + (
                    subcat_counts.get(subcat, 0) * subcat_weight
                )
                candidate_scores.append((cid, score))

            # Sort candidate items descending by affinity score
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            cat_ranked_ids = [cid for cid, score in candidate_scores]

        # --- B. DENSE COSINE RETRIEVAL ---
        user_vector = retriever.generate_user_vector(
            history_ids, retriever.embeddings_map
        )
        user_norm = np.linalg.norm(user_vector)
        if user_norm > 0:
            user_vector = user_vector / user_norm

        valid_dense_cids = [
            cid for cid in candidate_ids if cid in normalized_embeddings
        ]

        sorted_candidates = []
        if valid_dense_cids:
            # Vectorized Matrix Multiplication
            candidate_matrix = np.vstack(
                [normalized_embeddings[cid] for cid in valid_dense_cids]
            )
            scores = candidate_matrix @ user_vector
            top_indices = np.argsort(scores)[::-1]
            sorted_candidates = [valid_dense_cids[i] for i in top_indices]

        # --- C. RRF MERGING (Dense Cosine + Category Channel) ---
        rrf_scores = defaultdict(float)

        for rank, news_id in enumerate(cat_ranked_ids, start=1):
            rrf_scores[news_id] += 1.0 / (rrf_k + rank)

        for rank, news_id in enumerate(sorted_candidates, start=1):
            rrf_scores[news_id] += 1.0 / (rrf_k + rank)

        if not rrf_scores:
            continue

        final_ranked_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        # Compute evaluation metrics
        session_metric = calculate_metrics(
            final_ranked_ids, ground_truth_ids, k_list=k_list
        )
        if session_metric:
            all_session_metrics.append(session_metric)

    # 4. Aggregate metrics across all evaluated sessions
    if not all_session_metrics:
        return {}

    aggregated_metrics = {}
    metric_keys = all_session_metrics[0].keys()

    for key in metric_keys:
        aggregated_metrics[key] = float(
            np.mean([m[key] for m in all_session_metrics])
        )

    return aggregated_metrics

def evaluate_xgb_ranker_from_parquet(
    val_parquet_path,
    model,
    feature_cols,
    candidate_col="candidate_id",
    k_list=[10, 20],
):
    """Memory-efficient evaluation on Parquet feature files using NumPy vectorization."""
    # 1. Load ONLY required columns to minimize RAM usage
    required_cols = list(
        set(feature_cols + ["impression_id", candidate_col, "label"])
    )

    if isinstance(val_parquet_path, list):
        df_val = pd.concat(
            [
                pd.read_parquet(p, columns=required_cols)
                for p in val_parquet_path
            ],
            ignore_index=True,
        )
    else:
        df_val = pd.read_parquet(val_parquet_path, columns=required_cols)

    if df_val.empty:
        return {}

    # Sort in-place by impression_id to group contiguous blocks
    df_val.sort_values("impression_id", inplace=True)
    df_val.reset_index(drop=True, inplace=True)

    # 2. Batch Inference
    X_val = df_val[feature_cols].to_numpy(dtype=np.float32)

    if isinstance(model, xgb.Booster):
        dval = xgb.DMatrix(X_val, feature_names=feature_cols)
        df_val["score"] = model.predict(dval)
    else:
        df_val["score"] = model.predict(X_val)

    # Drop feature matrix from memory immediately
    del X_val
    if "dval" in locals():
        del dval

    # 3. Memory-Safe Evaluation Loop using Contiguous Block Slicing
    imp_ids = df_val["impression_id"].to_numpy()
    candidate_ids = df_val[candidate_col].to_numpy()
    labels = df_val["label"].to_numpy()
    scores = df_val["score"].to_numpy()

    # Find the boundaries for each impression_id without calling groupby
    _, start_indices, counts = np.unique(
        imp_ids, return_index=True, return_counts=True
    )

    all_session_metrics = []

    for start_idx, count in zip(start_indices, counts):
        end_idx = start_idx + count

        sess_scores = scores[start_idx:end_idx]
        sess_labels = labels[start_idx:end_idx]
        sess_cands = candidate_ids[start_idx:end_idx]

        # Skip impressions with all negative or all positive labels
        if not (sess_labels.any() and not sess_labels.all()):
            continue

        # Sort indices within this session by score descending
        order = np.argsort(-sess_scores)
        ranked_cands = sess_cands[order].tolist()

        # Extract positive ground truth IDs
        ground_truth_ids = set(sess_cands[sess_labels == 1].tolist())

        # Compute metric
        session_metric = calculate_metrics(
            ranked_cands, ground_truth_ids, k_list=k_list
        )
        if session_metric:
            all_session_metrics.append(session_metric)

    # 4. Aggregate Metrics
    if not all_session_metrics:
        return {}

    aggregated_metrics = {}
    for key in all_session_metrics[0].keys():
        aggregated_metrics[key] = float(
            np.mean([m[key] for m in all_session_metrics])
        )

    return aggregated_metrics
    