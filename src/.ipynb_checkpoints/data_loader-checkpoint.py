import pandas as pd
import gc
import numpy as np
import pandas as pd
import xgboost as xgb

def load_users_and_articles():
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
    df_news = pd.read_csv("news.tsv",
        sep="\t",
        header=None,
        names=news_cols,
        usecols=["news_id", "category", "subcategory", "title", "abstract"],
    )
    
    # Step 4: Parse Impression Behavior Logs
    behavior_cols = [
        "impression_id",
        "user_id",
        "time",
        "history",
        "impressions",
    ]
    df_behaviors = pd.read_csv("behaviors.tsv",
        sep="\t",
        header=None,
        names=behavior_cols,
    )
    
    # Drop rows without reading history
    df_behaviors = df_behaviors.dropna(subset=["history", "impressions"]).copy()
    
    print(f"Loaded {len(df_news):,} articles.")
    print(f"Loaded {len(df_behaviors):,} user impression sessions.")

    tsv_path = "behaviors.tsv"
    ground_truth_dict = {}
    
    # Stream the file line-by-line without loading it into Pandas
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
    
            impression_id = parts[0]
            impressions_str = parts[4]  # Column index 4 contains "N1234-1 N5678-0..."
    
            # Extract only items ending in '-1' (clicked articles)
            clicked_ids = {
                item[:-2]
                for item in impressions_str.split(" ")
                if item.endswith("-1")
            }
    
            if clicked_ids:
                ground_truth_dict[impression_id] = clicked_ids

    return df_news, df_behaviors, ground_truth_dict

# --- 2A. Filter-Enabled Streaming Iterator ---
class FilteredRankingIterator(xgb.DataIter):

  def __init__(self, file_paths, feature_cols, label_col, group_col, target_imps):
    self.file_paths = file_paths
    self.feature_cols = feature_cols
    self.label_col = label_col
    self.group_col = group_col
    self.target_imps = target_imps  # Set of allowed impression_ids for this split
    self._it = 0
    super().__init__()

  def next(self, input_data):
    while self._it < len(self.file_paths):
      file_path = self.file_paths[self._it]
      self._it += 1

      df = pd.read_parquet(file_path)

      # Filter strictly to impression_ids matching this temporal split
      df = df[df[self.group_col].astype(np.int32).isin(self.target_imps)]
      #print(len(df))
      if df.empty:
        continue

      # Group contiguous sessions
      df = df.sort_values(by=self.group_col)

      X = df[self.feature_cols].to_numpy(dtype=np.float32)
      y = df[self.label_col].to_numpy(dtype=np.int8)
      groups = (
          df.groupby(self.group_col, sort=False).size().to_numpy(dtype=np.int32)
      )

      input_data(data=X, label=y, group=groups)
      return 1

    return 0

  def reset(self):
    self._it = 0

class FastRankingIterator(xgb.DataIter):
    def __init__(self, file_paths, feature_cols, label_col="label", group_col="impression_id"):
        self.file_paths = file_paths
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.group_col = group_col
        self._it = 0
        super().__init__()

    def next(self, input_data):
        if self._it < len(self.file_paths):
            file_path = self.file_paths[self._it]
            self._it += 1
            
            df = pd.read_parquet(file_path)
            if df.empty:
                return 1

            # Ensure contiguous grouping per impression
            df.sort_values(by=self.group_col, inplace=True)

            X = df[self.feature_cols].to_numpy(dtype=np.float32)
            y = df[self.label_col].to_numpy(dtype=np.int8)
            groups = df.groupby(self.group_col, sort=False).size().to_numpy(dtype=np.int32)

            input_data(data=X, label=y, group=groups)
            return 1
        return 0

    def reset(self):
        self._it = 0