#%% Importing necessary libraries
import pandas as pd

#%% Display full columns in DataFrame output
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Prevent line wrapping
pd.set_option('display.max_rows', 10)       # Optional: control number of rows shown

#%%Read one file
df1 = pd.read_parquet('B2_raw_34S_19E_258N.parquet')
# Show shape of the DataFrame
print("Shape of first file:", df1.shape)

# Show head with all columns
print("Head of first file:")
print(df1.head())

#%%Read second file
df2 = pd.read_parquet('B2_raw_34S_19E_259N.parquet')

# Show shape of the DataFrame
print("\nShape of second file:", df2.shape)

# Show head with all columns
print("Head of second file:")
print(df2.head())

#%%
# Get the set of 'fid' values from both DataFrames
fids_1 = set(df1['fid'])
fids_2 = set(df2['fid'])

# Find the intersection (common fids)
common_fids = fids_1.intersection(fids_2)

# Check how many common fids there are
print(f"Number of common fids: {len(common_fids)}")

# Optionally, display the common fids
if common_fids:
    print("Some common fids:", list(common_fids)[:10])  # show first 10 common fids
else:
    print("No common fids found.")

#%%Read third parquet file
df3 = pd.read_parquet('B6_raw_34S_19E_258N.parquet')
print("Shape of third file:", df3.shape)
print("Head of third file:")
print(df3.head())

#%% Read fourth Parquet file
df4 = pd.read_parquet('B6_raw_34S_19E_259N.parquet')
print("\nShape of fourth file:", df4.shape)
print("Head of fourth file:")
print(df4.head())

#%%
# Get unique fids from file 1 and 2
fids_12 = set(df1['fid']).union(set(df2['fid']))

# Get unique fids from file 3 and 4
fids_34 = set(df3['fid']).union(set(df4['fid']))

# Find common fids between group (1 & 2) and group (3 & 4)
common_fids_12_34 = fids_12.intersection(fids_34)

# Show result
print(f"Number of common fids between files 1&2 and files 3&4: {len(common_fids_12_34)}")

# Optionally view some of them
if common_fids_12_34:
    print("Some common fids:", list(common_fids_12_34)[:10])  # show first 10
else:
    print("No common fids found.")

#%%Read fifth parquet file
df5 = pd.read_parquet('hue_raw_34S_19E_258N.parquet')
print("\nShape of fourth file:", df5.shape)
print("Head of fourth file:")
print(df5.head())

#%%Read sixth parquet file
df6 = pd.read_parquet('hue_raw_34S_19E_259N.parquet')
print("\nShape of fourth file:", df6.shape)
print("Head of fourth file:")
print(df6.head())

#%%
df7 = pd.read_parquet('EVI_raw_34S_19E_258N.parquet')
print("\nShape of fourth file:", df7.shape)
print("Head of fourth file:")
print(df7.head())

#%%
df8 = pd.read_parquet('EVI_raw_34S_19E_259N.parquet')
print("\nShape of fourth file:", df8.shape)
print("Head of fourth file:")
print(df8.head())

#%%
df9 = pd.read_parquet('B11_raw_34S_19E_258N.parquet')
print("\nShape of fourth file:", df9.shape)
print("Head of fourth file:")
print(df9.head())

#%%
df10 = pd.read_parquet('B11_raw_34S_19E_259N.parquet')
print("\nShape of fourth file:", df10.shape)
print("Head of fourth file:")
print(df10.head())

#%%
df11 = pd.read_parquet('B12_raw_34S_19E_258N.parquet')
print("\nShape of fourth file:", df11.shape)
print("Head of fourth file:")
print(df11.head())

#%%
df12 = pd.read_parquet('B12_raw_34S_19E_259N.parquet')
print("\nShape of fourth file:", df12.shape)
print("Head of fourth file:")
print(df12.head())

#%% Common 258
dfs_258 = [df1, df3, df5, df7, df9,df11]  # Replace with your actual list of DataFrames

# Extract sets of fids
fid_sets = [set(df['fid']) for df in dfs_258]

# Find intersection across all
common_fids = set.intersection(*fid_sets)

print(f"Number of common fids: {len(common_fids)}")
print("Some common fids:", list(common_fids)[:10])

#%% Common 259
dfs_259 = [df2, df4, df6, df8, df10,df12]  # Replace with your actual list of DataFrames

# Extract sets of fids
fid_sets = [set(df['fid']) for df in dfs_259]

# Find intersection across all
common_fids = set.intersection(*fid_sets)

print(f"Number of common fids: {len(common_fids)}")
print("Some common fids:", list(common_fids)[:10])

#%% Common 258&259
dfs_258_259 = [df1,df2,df3, df4,df5, df6,df7, df8,df9, df10,df11,df12]  # Replace with your actual list of DataFrames

# Extract sets of fids
fid_sets = [set(df['fid']) for df in dfs_258_259]

# Find intersection across all
common_fids = set.intersection(*fid_sets)

print(f"Number of common fids: {len(common_fids)}")
print("Some common fids:", list(common_fids)[:10])

#%%
dfs_258 = [df1, df3, df5, df7, df9, df11]  # Your list of DataFrames

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs_258, ignore_index=True)

unique_fid_count = combined_df['fid'].nunique()

print(f"Number of unique fids: {unique_fid_count}")

#%%
import pandas as pd

#%% Display full columns in DataFrame output
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Prevent line wrapping
pd.set_option('display.max_rows', 10)       # Optional: control number of rows shown

#%%Read one file
df_258 = pd.read_parquet('dl_258N.parquet')
# Show shape of the DataFrame
print("Shape of first file:", df_258.shape)

# Show head with all columns
print("Head of first file:")
print(df_258.head())

#%%
df_259 = pd.read_parquet('dl_259N.parquet')
# Show shape of the DataFrame
print("Shape of first file:", df_259.shape)

# Show head with all columns
print("Head of first file:")
print(df_259.head())

#%%
merged_df = pd.concat([df_258, df_259], ignore_index=True)

# Show shape and preview of merged DataFrame
print("Shape of merged DataFrame:", merged_df.shape)
print("Head of merged DataFrame:")
print(merged_df.head())

#%%

print(merged_df.isna().sum().to_string())

#%%
cols_to_drop = [col for col in merged_df.columns if col.endswith('2017_05')]
merged_df = merged_df.drop(columns=cols_to_drop)

print(f"✅ Dropped {len(cols_to_drop)} columns ending with '2017_05'")

#%%
print(merged_df.isna().sum().to_string())

#%% Save to new Parquet file
merged_df.to_parquet('merged_dl_258_259.parquet', index=False)
print("✅ Merged DataFrame saved as 'merged_dl_258_259.parquet'")

#%%
print(merged_df['fid'].nunique())

#%%






