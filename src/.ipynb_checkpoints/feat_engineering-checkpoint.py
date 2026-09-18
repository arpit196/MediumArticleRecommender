from collections import defaultdict
import pandas as pd

def compute_expanding_window_ctr(
    df_behaviors, smoothing=10.0, global_prior=0.03
):
    """Computes leak-free historical CTR for each impression row based strictly

    on clicks observed BEFORE that timestamp.
    """
    # 1. Sort behaviors chronologically
    df_sorted = df_behaviors.sort_values("time").reset_index(drop=True)

    clicks = defaultdict(int)
    impressions = defaultdict(int)

    row_ctr_features = []

    for row in df_sorted.itertuples(index=False):
        imp_str = getattr(row, "impressions", "")
        if not isinstance(imp_str, str) or not imp_str:
            row_ctr_features.append({})
            continue

        items = imp_str.split(" ")

        # A. FIRST extract feature using ONLY PAST counts (Leak-free)
        current_row_ctrs = {}
        for item in items:
            if "-" not in item:
                continue
            news_id, _ = item.rsplit("-", 1)

            if news_id not in current_row_ctrs:
                c = clicks[news_id]
                total_imps = impressions[news_id]
                # Bayesian smoothed CTR using past observations only
                smoothed_ctr = (c + smoothing * global_prior) / (
                    total_imps + smoothing
                )
                current_row_ctrs[news_id] = float(smoothed_ctr)

        row_ctr_features.append(current_row_ctrs)

        # B. THEN update state with current row's outcomes for FUTURE rows to use
        for item in items:
            if "-" not in item:
                continue
            news_id, label = item.rsplit("-", 1)
            impressions[news_id] += 1
            if label == "1":
                clicks[news_id] += 1

    # Final state dictionary can be used directly on Validation/Test sets
    final_historical_ctr_map = {
        nid: (clicks[nid] + smoothing * global_prior)
        / (impressions[nid] + smoothing)
        for nid in impressions
    }

    df_sorted["leak_free_ctr_dict"] = row_ctr_features
    return df_sorted, final_historical_ctr_map


def extract_ranking_features_batched(
    df_behaviors,
    df_news,
    retriever,
    global_ctr_map=None,
    batch_size=10000,  # Reduced batch size to manage candidate row explosion
    output_dir="./feature_chunksx",
    return_df_in_memory=False,  # Set to False to keep memory safe
):
    """Extracts candidate and user-level ranking features using memory-safe batching.

    Processes user behavior logs and candidate news impressions to generate candidate
    similarity, category affinity, diversity, and positional features. Implements
    LRU caching for user profiles and explicit downcasting to manage memory usage, 
    streaming outputs directly to Parquet chunks on disk.

    Args:
        df_behaviors (pd.DataFrame): User interaction logs containing 'history' and 
            'impressions' columns.
        df_news (pd.DataFrame): News metadata containing 'news_id', 'category', and 
            'subcategory'.
        retriever (HybridRetriever): Retriever instance containing pre-computed 
            embeddings mapped via `retriever.embeddings_map`.
        global_ctr_map (dict, optional): Historical Click-Through Rate mapping structured 
            as `{impression_id: {candidate_id: ctr_float}}`. Defaults to None.
        batch_size (int, optional): Number of behavior rows processed per batch. 
            Defaults to 10000.
        output_dir (str, optional): Target directory path to save Parquet chunks. 
            Defaults to "./feature_chunksx".
        return_df_in_memory (bool, optional): If True, loads and concatenates all chunks 
            into a single memory DataFrame. Defaults to False.

    Returns:
        tuple: A 2-element tuple containing:
            - chunk_files (list[str]) or df_features (pd.DataFrame): List of saved 
              Parquet file paths if `return_df_in_memory=False`, otherwise the 
              concatenated Pandas DataFrame.
            - group_ids (list[int]): Candidate count per impression session, formatted 
              for Learning-to-Rank models (e.g., LightGBM / XGBoost `group` query arrays).

    Extracted Features:
        - impression_id (str): Unique identifier for the impression session.
        - candidate_id (str): Identifier for the candidate news article.
        - label (int8): Ground truth binary label (1 for click, 0 otherwise).
        - dense_cosine_sim (float32): Cosine similarity between candidate and average 
          user profile embedding.
        - recency_weighted_sim (float32): Cosine similarity weighted exponentially 
          toward the user's most recent reading history.
        - max_cosine_sim (float32): Highest cosine similarity score among all individual 
          articles in user history.
        - top3_sim (float32): Average similarity score across the top 3 most similar 
          historical articles.
        - category_match (float32): Binary indicator (1.0/0.0) if candidate category 
          matches user's most frequent category.
        - user_cat_affinity (float32): Proportion of user history spent reading candidate's 
          category.
        - subcategory_match (float32): Binary indicator (1.0/0.0) matching candidate 
          subcategory to user's top subcategory.
        - user_subcat_affinity (float32): Proportion of user history spent reading 
          candidate's subcategory.
        - is_new_category (float32): Binary indicator (1.0/0.0) if user has never 
          interacted with candidate's category.
        - user_hist_len (int16): Total count of articles in user reading history.
        - user_history_entropy (float32): Entropy score measuring category diversity in 
          user reading profile.
        - impression_position (int16): Zero-indexed absolute position of candidate in 
          impression list.
        - normalized_imp_position (float32): Relative impression position scaled [0.0, 1.0].
        - global_ctr (float32): Pre-calculated historical Click-Through Rate for candidate article.
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    print("🚀 Starting SOTA Batched Feature Extraction...")

    # 1. Lookups & CTR Map
    news_cat_map = dict(zip(df_news["news_id"], df_news["category"].fillna("")))
    news_subcat_map = dict(
        zip(df_news["news_id"], df_news["subcategory"].fillna(""))
    )
    ctr_map = global_ctr_map or {}

    # 2. Pre-normalize embeddings
    normalized_embeddings = {}
    for news_id, vec in retriever.embeddings_map.items():
        norm = np.linalg.norm(vec)
        normalized_embeddings[news_id] = (
            (vec / (norm + 1e-9)).astype(np.float32)
            if norm > 0
            else vec.astype(np.float32)
        )

    # 3. LRU Cache for User Profiles (Limits RAM retention to last 10,000 users)
    @lru_cache(maxsize=10000)
    def get_user_profile(history_tuple):
        history_ids = list(history_tuple)
        u_hist_len = len(history_ids)

        # Categories
        cat_counts = Counter(
            news_cat_map[hid] for hid in history_ids if hid in news_cat_map
        )
        subcat_counts = Counter(
            news_subcat_map[hid] for hid in history_ids if hid in news_subcat_map
        )

        total_cats = sum(cat_counts.values())
        total_subcats = sum(subcat_counts.values())

        if total_cats > 0:
            cat_affinity_dict = {
                c: cnt / total_cats for c, cnt in cat_counts.items()
            }
            top_user_cat = cat_counts.most_common(1)[0][0]
            probs = np.array(list(cat_affinity_dict.values()))
            user_entropy = -float(np.sum(probs * np.log2(probs + 1e-9)))
        else:
            cat_affinity_dict, top_user_cat, user_entropy = {}, "", 0.0

        if total_subcats > 0:
            subcat_affinity_dict = {
                sc: cnt / total_subcats for sc, cnt in subcat_counts.items()
            }
            top_user_subcat = subcat_counts.most_common(1)[0][0]
        else:
            subcat_affinity_dict, top_user_subcat = {}, ""

        # Dense Vectors
        hist_vectors = [
            normalized_embeddings[hid]
            for hid in history_ids
            if hid in normalized_embeddings
        ]

        if hist_vectors:
            hist_matrix = np.array(hist_vectors, dtype=np.float32)

            user_vec = np.mean(hist_matrix, axis=0)
            u_norm = np.linalg.norm(user_vec)
            user_vec = (
                user_vec / u_norm if u_norm > 0 else user_vec
            ).astype(np.float32)

            weights = np.exp(np.linspace(-1.0, 0.0, len(hist_vectors))).astype(
                np.float32
            )
            weights /= np.sum(weights)
            recency_user_vec = np.sum(
                hist_matrix * weights[:, None], axis=0
            ).astype(np.float32)
            r_norm = np.linalg.norm(recency_user_vec)
            recency_user_vec = (
                recency_user_vec / r_norm if r_norm > 0 else recency_user_vec
            ).astype(np.float32)
        else:
            hist_matrix = None
            user_vec = np.zeros(384, dtype=np.float32)
            recency_user_vec = np.zeros(384, dtype=np.float32)

        return (
            u_hist_len,
            cat_affinity_dict,
            top_user_cat,
            subcat_affinity_dict,
            top_user_subcat,
            user_entropy,
            user_vec,
            recency_user_vec,
            hist_matrix,
        )

    total_rows = len(df_behaviors)
    num_batches = int(np.ceil(total_rows / batch_size))
    chunk_files = []
    group_ids = []

    for batch_idx in range(num_batches):
        print(
            f"\n📦 Processing Batch {batch_idx + 1}/{num_batches} (Rows {batch_idx*batch_size} to {min((batch_idx+1)*batch_size, total_rows)})..."
        )

        df_batch = df_behaviors.iloc[
            batch_idx * batch_size : (batch_idx + 1) * batch_size
        ]
        records = []

        for row in tqdm(
            df_batch.itertuples(index=False),
            total=len(df_batch),
            desc=f"Batch {batch_idx+1}",
        ):
            imp_id = str(
                getattr(
                    row,
                    "impression_id",
                    getattr(row, "Impression_ID", row[0]),
                )
            )
            history_ids = (
                row.history.split(" ") if isinstance(row.history, str) else []
            )
            impression_items = row.impressions.split(" ")

            candidate_ids, c_labels = [], []
            for item in impression_items:
                parts = item.rsplit("-", 1)
                candidate_ids.append(parts[0])
                c_labels.append(int(parts[1]) if len(parts) > 1 else 0)

            (
                u_hist_len,
                cat_affinity_dict,
                top_user_cat,
                subcat_affinity_dict,
                top_user_subcat,
                user_entropy,
                user_vec,
                recency_user_vec,
                hist_matrix,
            ) = get_user_profile(tuple(history_ids))

            session_candidates_count = 0
            total_imp_candidates = len(candidate_ids)

            for pos, (cid, label) in enumerate(
                zip(candidate_ids, c_labels)
            ):
                if cid not in news_cat_map:
                    continue

                cand_cat = news_cat_map[cid]
                cand_subcat = news_subcat_map[cid]
                c_vec = normalized_embeddings.get(cid, None)

                # Similarity
                if c_vec is not None:
                    dense_sim = float(np.dot(c_vec, user_vec))
                    recency_sim = float(np.dot(c_vec, recency_user_vec))
                    max_sim = (
                        float(np.max(np.dot(hist_matrix, c_vec)))
                        if hist_matrix is not None
                        else 0.0
                    )
                else:
                    dense_sim, recency_sim, max_sim = 0.0, 0.0, 0.0

                # Categories
                cat_affinity = cat_affinity_dict.get(cand_cat, 0.0)
                subcat_affinity = subcat_affinity_dict.get(cand_subcat, 0.0)
                if hist_matrix is not None and c_vec is not None:
                    all_sims = np.dot(hist_matrix, c_vec)
                    #max_sim = float(np.max(all_sims))
                    top3_sim = float(np.mean(np.sort(all_sims)[-3:])) # Top-3 historical items average
                else:
                    top3_sim = 0.0, 0.0
                # Append compact record dictionary
                records.append(
                    {
                        "impression_id": imp_id,
                        "candidate_id": cid,
                        "label": np.int8(label),
                        "dense_cosine_sim": np.float32(dense_sim),
                        "max_cosine_sim": np.float32(max_sim),
                        "recency_weighted_sim": np.float32(recency_sim),
                        "category_match": np.float32(
                            1.0 if cand_cat == top_user_cat else 0.0
                        ),
                        "user_cat_affinity": np.float32(cat_affinity),
                        "subcategory_match": np.float32(
                            1.0 if cand_subcat == top_user_subcat else 0.0
                        ),
                        "top3_sim": np.float32(top3_sim),
                        "user_subcat_affinity": np.float32(subcat_affinity),
                        "is_new_category": np.float32(
                            1.0 if cand_cat not in cat_affinity_dict else 0.0
                        ),
                        "user_hist_len": np.int16(u_hist_len),
                        "user_history_entropy": np.float32(user_entropy),
                        "impression_position": np.int16(pos),
                        "normalized_imp_position": np.float32(
                            pos / total_imp_candidates
                            if total_imp_candidates > 0
                            else 0.0
                        ),
                        "global_ctr": np.float32(ctr_map.get(imp_id, {}).get(cid, 0.0)),
                    }
                )

                session_candidates_count += 1

            if session_candidates_count > 0:
                group_ids.append(session_candidates_count)

        # Convert records to DataFrame & export chunk
        df_chunk = pd.DataFrame(records)
        chunk_path = os.path.join(output_dir, f"chunk_{batch_idx}.parquet")
        df_chunk.to_parquet(chunk_path, index=False, compression="snappy")
        chunk_files.append(chunk_path)

        # Clear batch structures and enforce garbage collection
        del df_chunk, records
        gc.collect()

    print(
        f"\n✅ Extraction complete in {time.time() - start_time:.2f}s! Chunks saved to '{output_dir}'."
    )

    if return_df_in_memory:
        df_features = pd.concat(
            [pd.read_parquet(f) for f in chunk_files], ignore_index=True
        )
        return df_features, group_ids

    return chunk_files, group_ids