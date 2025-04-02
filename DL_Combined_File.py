import pandas as pd

# Standard column names
standard_columns = [
    'id', 'point', 'fid', 'SHAPE_AREA', 'SHAPE_LEN', 'crop_id', 'crop_name',
    'January', 'February', 'March', 'April', 'May', 'July', 'August',
    'September', 'October', 'November', 'December'
]

# Function to load, rename, and tag file with Type
def load_and_prepare(file_path, type_label):
    df = pd.read_parquet(file_path)

    # Check if the number of columns matches
    if len(df.columns) != len(standard_columns):
        raise ValueError(f"Column count mismatch in {file_path}. Expected {len(standard_columns)}, found {len(df.columns)}.")

    df.columns = standard_columns
    df['Type'] = type_label
    return df

# File metadata: (file_path, type_label)
file_info = [
    ('B2_raw_34S_19E_258N.parquet', 'B2'), ('B2_raw_34S_19E_259N.parquet', 'B2'),
    ('B6_raw_34S_19E_258N.parquet', 'B6'), ('B6_raw_34S_19E_259N.parquet', 'B6'),
    ('B11_raw_34S_19E_258N.parquet', 'B11'), ('B11_raw_34S_19E_259N.parquet', 'B11'),
    ('B12_raw_34S_19E_258N.parquet', 'B12'), ('B12_raw_34S_19E_259N.parquet', 'B12'),
    ('EVI_raw_34S_19E_258N.parquet', 'EVI'), ('EVI_raw_34S_19E_259N.parquet', 'EVI'),
    ('hue_raw_34S_19E_258N.parquet', 'HUE'), ('hue_raw_34S_19E_259N.parquet', 'HUE')
]

# Load and prepare all files
dataframes = [load_and_prepare(file, label) for file, label in file_info]

# Combine into one DataFrame
combined_df = pd.concat(dataframes, axis=0).reset_index(drop=True)

# Done!
print("Combined shape:", combined_df.shape)
print(combined_df.head())

#%%
combined_df.to_parquet("merged_DL_data.parquet", index=False)

print("Combined data saved as 'merged_DL_data.parquet'")

#%%
print(combined_df['Type'].unique())
