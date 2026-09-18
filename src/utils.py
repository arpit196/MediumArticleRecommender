import pyarrow.dataset as ds
import pandas as pd
def get_group_ids_from_parquet(parquet_path):
    """Reads only the impression_id column from disk to compute ranking group lengths."""
    dataset = ds.dataset(parquet_path, format="parquet")

    # Read ONLY the impression_id column to conserve RAM
    table = dataset.to_table(columns=["impression_id"])

    # Compute group sizes efficiently via PyArrow value counts
    counts_df = table.to_pandas().groupby("impression_id", sort=False).size()
    return counts_df.values.tolist()