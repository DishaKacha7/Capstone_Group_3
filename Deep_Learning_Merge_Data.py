import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def load_and_merge_parquet_for_dl(directory):
    """
    Load and merge all parquet files from directory based on 'fid' and 'point'.
    Rename feature columns with band prefix, return merged DataFrame.
    """
    parquet_files = [f for f in os.listdir(directory) if f.endswith('.parquet')]

    if not parquet_files:
        print(f"No Parquet files found in {directory}")
        return None

    merged_df = None
    for file in parquet_files:
        file_path = os.path.join(directory, file)
        df = pd.read_parquet(file_path)

        band = file.split('_')[0]
        base_cols = ['fid', 'point']
        meta_cols = ['crop_id', 'crop_name']

        # Select relevant columns
        band_cols = [col for col in df.columns if '/' in col or band in col]
        band_cols = base_cols + meta_cols + band_cols
        df = df[band_cols]

        # Rename band columns
        ts_cols = [col for col in df.columns if col not in base_cols + meta_cols]
        df = df.rename(columns={col: f"{band}_{col}" for col in ts_cols})

        # Merge on fid, point, crop_id, crop_name
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(
                merged_df,
                df,
                on=['fid', 'point', 'crop_id', 'crop_name'],
                how='outer'
            )

    print(f"\nMerged DataFrame shape: {merged_df.shape}")
    print("\nPreview of merged DataFrame:")
    print(merged_df.head(10))

    return merged_df


if __name__ == "__main__":
    data_dir = '/home/ubuntu/Capstone_AWS'  # Change this to your actual path

    # Step 1: Load and merge files without any NaN removal
    merged_df = load_and_merge_parquet_for_dl(data_dir)

    # Step 2: Just preview the merged DataFrame
    if merged_df is not None:
        print("\nMerged data loaded successfully.")
