import pandas as pd
import numpy as np
import os


def load_and_merge_parquet_files(directory):
    """
    Load all parquet files from directory and merge them based on common columns
    """
    # Get all parquet files in directory
    parquet_files = [f for f in os.listdir(directory) if f.endswith('259N.parquet')]

    merged_df = None
    for file in parquet_files:
        # Load parquet file
        file_path = os.path.join(directory, file)
        df = pd.read_parquet(file_path)

        # Get band name from file (e.g., 'B11', 'EVI', etc.)
        band = file.split('_')[0]

        # Rename numerical columns to include band name
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        rename_dict = {col: f"{band}_{col}" for col in numeric_cols
                      if not col in ['id', 'point', 'fid', 'crop_id', 'SHAPE_AREA', 'SHAPE_LEN']}
        df = df.rename(columns=rename_dict)

        if merged_df is None:
            # Keep all columns for first file
            merged_df = df
        else:
            # Only keep numeric columns (with renamed band prefix) and merge
            cols_to_keep = [col for col in df.columns
                          if col.startswith(f"{band}_") or
                          col in ['id', 'point', 'fid']]
            df = df[cols_to_keep]
            merged_df = pd.merge(merged_df, df, on=['id', 'point', 'fid'])

    merged_df.to_parquet("dl_259N.parquet", index=False)
    print("The final data was saved in the csv.!!!!!!!")
    return merged_df


merged_df = load_and_merge_parquet_files("./")
