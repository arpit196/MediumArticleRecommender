import argparse
import glob
import json
import os
import pickle
import faiss
import pandas as pd
import xgboost as xgb

# Core module imports from src package
from src.data_loader import FastRankingIterator, load_users_and_articles
from src.eval import evaluate_impression_pipeline, evaluate_xgb_ranker_from_parquet
from src.feature_engineering import extract_ranking_features_batched
from src.retriever import HybridRetriever


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost Ranker Model")
    parser.add_argument(
        "--extract-features",
        action="store_true",
        help="If set, runs batch feature extraction before training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("🚀 Starting Training & Pipeline Execution...")

    # 1. Load Data & Ground Truth Maps
    df_news, df_behaviors, ground_truth_dict = load_users_and_articles()

    # Load pre-computed CTR mapping if available
    ctr_map = None
    if os.path.exists("impression_ctr_map.json"):
        with open("impression_ctr_map.json", "r") as f:
            ctr_map = json.load(f)

    # 2. Load Retriever & Vector Index Assets
    retriever = HybridRetriever()

    index_filename = "faiss_hnsw_index.bin"
    if os.path.exists(index_filename):
        retriever.index = faiss.read_index(index_filename)

    embeddings_map_filename = "embeddings_map.pkl"
    if os.path.exists(embeddings_map_filename):
        with open(embeddings_map_filename, "rb") as f:
            retriever.embeddings_map = pickle.load(f)

    # 3. Conditional Feature Extraction
    if args.extract_features:
        print("\n⚙️ Running batch feature extraction...")
        df_train_feats, train_groups = extract_ranking_features_batched(
            df_behaviors,
            df_news,
            retriever,
            global_ctr_map=ctr_map,
            output_dir="./train_chunks",
        )
        print("✅ Feature extraction complete!")
    else:
        print("\n⏩ Skipping feature extraction (using existing chunks on disk)...")

    # 4. Initialize Streaming QuantileDMatrix
    print("\n📦 Initializing QuantileDMatrix from feature chunks...")
    dtrain, dval, feature_cols, val_chunk_files = create_train_val_split()

    # 5. Train XGBRanker Model
    print("\n🎯 Training XGBoost Ranker Model...")
    params = {
        "objective": "rank:ndcg",
        "eval_metric": ["ndcg@5", "ndcg@10", "map"],
        "tree_method": "hist",
        "learning_rate": 0.02,
        "max_depth": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 50,
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dval, "val")],
        num_boost_round=100,
    )
    print("✅ Model training complete!")


if __name__ == "__main__":
    main()