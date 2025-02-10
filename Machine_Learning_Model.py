#%%Importing necessary libraries
import pandas as pd

#%% File 1 HUE
file_1 = "hue_34S_19E_258N.parquet"
df1 = pd.read_parquet(file_1)
pd.set_option("display.max_columns", None)
print(df1.head(10))

#%%
num_rows, num_columns = df1.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df1.columns = [
    "id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name",
    "ABS_Energy", "Absolute_Sum_of_Changes", "Autocorr_lag_1", "Autocorr_lag_2", "Autocorr_lag_3",
    "Count_above_mean", "Count_below_mean", "Doy_of_Maximum_dates", "Doy_of_Minimum_dates",
    "Kurtosis", "Large_Standard_Deviation", "Maximum", "Mean", "Mean_abs_Change",
    "Mean_Change", "Mean_Second_Derivative_Central", "Median", "Minimum",
    "Quantile_pointzerofive", "Quantile_pointninefive", "Ratio_Beyond_Sigma_R1",
    "Ratio_Beyond_Sigma_R2", "Skewness", "Standard_Deviation", "Sum", "Symmetry_Looking",
    "TS_Complexity_Cid_ce", "Variance", "Variance_Larger_than_standard_deviation"
]

#%%
print(df1.head(10))
num_rows, num_columns = df1.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df1['Type'] = 'HUE'
print(df1.head(10))

#%% File 2 HUE
file_2 = "hue_34S_19E_259N.parquet"
df2 = pd.read_parquet(file_2)
pd.set_option("display.max_columns", None)
print(df2.head(10))

#%%
num_rows, num_columns = df2.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df2.columns = [
    "id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name",
    "ABS_Energy", "Absolute_Sum_of_Changes", "Autocorr_lag_1", "Autocorr_lag_2", "Autocorr_lag_3",
    "Count_above_mean", "Count_below_mean", "Doy_of_Maximum_dates", "Doy_of_Minimum_dates",
    "Kurtosis", "Large_Standard_Deviation", "Maximum", "Mean", "Mean_abs_Change",
    "Mean_Change", "Mean_Second_Derivative_Central", "Median", "Minimum",
    "Quantile_pointzerofive", "Quantile_pointninefive", "Ratio_Beyond_Sigma_R1",
    "Ratio_Beyond_Sigma_R2", "Skewness", "Standard_Deviation", "Sum", "Symmetry_Looking",
    "TS_Complexity_Cid_ce", "Variance", "Variance_Larger_than_standard_deviation"
]

#%%
print(df2.head(10))
num_rows, num_columns = df2.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df2['Type'] = 'HUE'
print(df2.head(10))

#%% File 3 EVI
file_3 = "EVI_34S_19E_258N.parquet"
df3 = pd.read_parquet(file_3)
pd.set_option("display.max_columns", None)
print(df3.head(10))

#%%
num_rows, num_columns = df3.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
columns_to_drop = [
    "B11_large_standard_deviation", "B11_standard_deviation", "B11_variance_larger_than_standard_deviation",
    "B12_large_standard_deviation", "B12_standard_deviation", "B12_variance_larger_than_standard_deviation",
    "B2_large_standard_deviation", "B2_standard_deviation", "B2_variance_larger_than_standard_deviation",
    "B6_large_standard_deviation", "B6_standard_deviation", "B6_variance_larger_than_standard_deviation",
    "hue_large_standard_deviation", "hue_standard_deviation", "hue_variance_larger_than_standard_deviation"
]

df3 = df3.drop(columns=[col for col in columns_to_drop if col in df3.columns])
#%%
num_rows, num_columns = df3.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df3.columns = [
    "id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name",
    "ABS_Energy", "Absolute_Sum_of_Changes", "Autocorr_lag_1", "Autocorr_lag_2", "Autocorr_lag_3",
    "Count_above_mean", "Count_below_mean", "Doy_of_Maximum_dates", "Doy_of_Minimum_dates",
    "Kurtosis", "Large_Standard_Deviation", "Maximum", "Mean", "Mean_abs_Change",
    "Mean_Change", "Mean_Second_Derivative_Central", "Median", "Minimum",
    "Quantile_pointzerofive", "Quantile_pointninefive", "Ratio_Beyond_Sigma_R1",
    "Ratio_Beyond_Sigma_R2", "Skewness", "Standard_Deviation", "Sum", "Symmetry_Looking",
    "TS_Complexity_Cid_ce", "Variance", "Variance_Larger_than_standard_deviation"
]

#%%
print(df3.head(10))
num_rows, num_columns = df3.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df3['Type'] = 'EVI'
print(df3.head(10))

#%% File 4 EVI
file_4 = "EVI_34S_19E_259N.parquet"
df4 = pd.read_parquet(file_4)
pd.set_option("display.max_columns", None)
print(df4.head(10))

#%%
num_rows, num_columns = df4.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
columns_to_drop = [
    "B11_large_standard_deviation", "B11_standard_deviation", "B11_variance_larger_than_standard_deviation",
    "B12_large_standard_deviation", "B12_standard_deviation", "B12_variance_larger_than_standard_deviation",
    "B2_large_standard_deviation", "B2_standard_deviation", "B2_variance_larger_than_standard_deviation",
    "B6_large_standard_deviation", "B6_standard_deviation", "B6_variance_larger_than_standard_deviation",
    "hue_large_standard_deviation", "hue_standard_deviation", "hue_variance_larger_than_standard_deviation"
]

df4 = df4.drop(columns=[col for col in columns_to_drop if col in df4.columns])
#%%
num_rows, num_columns = df4.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df4.columns = [
    "id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name",
    "ABS_Energy", "Absolute_Sum_of_Changes", "Autocorr_lag_1", "Autocorr_lag_2", "Autocorr_lag_3",
    "Count_above_mean", "Count_below_mean", "Doy_of_Maximum_dates", "Doy_of_Minimum_dates",
    "Kurtosis", "Large_Standard_Deviation", "Maximum", "Mean", "Mean_abs_Change",
    "Mean_Change", "Mean_Second_Derivative_Central", "Median", "Minimum",
    "Quantile_pointzerofive", "Quantile_pointninefive", "Ratio_Beyond_Sigma_R1",
    "Ratio_Beyond_Sigma_R2", "Skewness", "Standard_Deviation", "Sum", "Symmetry_Looking",
    "TS_Complexity_Cid_ce", "Variance", "Variance_Larger_than_standard_deviation"
]

#%%
print(df4.head(10))
num_rows, num_columns = df4.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df4['Type'] = 'EVI'
print(df4.head(10))

#%% File 5 B2
file_5 = "B2_34S_19E_258N.parquet"
df5 = pd.read_parquet(file_5)
pd.set_option("display.max_columns", None)
print(df5.head(10))

#%%
num_rows, num_columns = df5.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df5.columns = [
    "id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name",
    "ABS_Energy", "Absolute_Sum_of_Changes", "Autocorr_lag_1", "Autocorr_lag_2", "Autocorr_lag_3",
    "Count_above_mean", "Count_below_mean", "Doy_of_Maximum_dates", "Doy_of_Minimum_dates",
    "Kurtosis", "Large_Standard_Deviation", "Maximum", "Mean", "Mean_abs_Change",
    "Mean_Change", "Mean_Second_Derivative_Central", "Median", "Minimum",
    "Quantile_pointzerofive", "Quantile_pointninefive", "Ratio_Beyond_Sigma_R1",
    "Ratio_Beyond_Sigma_R2", "Skewness", "Standard_Deviation", "Sum", "Symmetry_Looking",
    "TS_Complexity_Cid_ce", "Variance", "Variance_Larger_than_standard_deviation"
]

#%%
print(df5.head(10))
num_rows, num_columns = df5.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df5['Type'] = 'B2'
print(df5.head(10))

#%% File 6 B2
file_6 = "B2_34S_19E_259N.parquet"
df6 = pd.read_parquet(file_6)
pd.set_option("display.max_columns", None)
print(df6.head(10))

#%%
num_rows, num_columns = df6.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df6.columns = [
    "id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name",
    "ABS_Energy", "Absolute_Sum_of_Changes", "Autocorr_lag_1", "Autocorr_lag_2", "Autocorr_lag_3",
    "Count_above_mean", "Count_below_mean", "Doy_of_Maximum_dates", "Doy_of_Minimum_dates",
    "Kurtosis", "Large_Standard_Deviation", "Maximum", "Mean", "Mean_abs_Change",
    "Mean_Change", "Mean_Second_Derivative_Central", "Median", "Minimum",
    "Quantile_pointzerofive", "Quantile_pointninefive", "Ratio_Beyond_Sigma_R1",
    "Ratio_Beyond_Sigma_R2", "Skewness", "Standard_Deviation", "Sum", "Symmetry_Looking",
    "TS_Complexity_Cid_ce", "Variance", "Variance_Larger_than_standard_deviation"
]

#%%
print(df6.head(10))
num_rows, num_columns = df6.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df6['Type'] = 'B2'
print(df6.head(10))

#%% File 7 B6
file_7 = "B6_34S_19E_258N.parquet"
df7 = pd.read_parquet(file_7)
pd.set_option("display.max_columns", None)
print(df7.head(10))

#%%
num_rows, num_columns = df7.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df7.columns = [
    "id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name",
    "ABS_Energy", "Absolute_Sum_of_Changes", "Autocorr_lag_1", "Autocorr_lag_2", "Autocorr_lag_3",
    "Count_above_mean", "Count_below_mean", "Doy_of_Maximum_dates", "Doy_of_Minimum_dates",
    "Kurtosis", "Large_Standard_Deviation", "Maximum", "Mean", "Mean_abs_Change",
    "Mean_Change", "Mean_Second_Derivative_Central", "Median", "Minimum",
    "Quantile_pointzerofive", "Quantile_pointninefive", "Ratio_Beyond_Sigma_R1",
    "Ratio_Beyond_Sigma_R2", "Skewness", "Standard_Deviation", "Sum", "Symmetry_Looking",
    "TS_Complexity_Cid_ce", "Variance", "Variance_Larger_than_standard_deviation"
]

#%%
print(df7.head(10))
num_rows, num_columns = df7.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df7['Type'] = 'B6'
print(df7.head(10))

#%% File 8 B6
file_8 = "B6_34S_19E_259N.parquet"
df8 = pd.read_parquet(file_8)
pd.set_option("display.max_columns", None)
print(df8.head(10))

#%%
num_rows, num_columns = df8.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df8.columns = [
    "id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name",
    "ABS_Energy", "Absolute_Sum_of_Changes", "Autocorr_lag_1", "Autocorr_lag_2", "Autocorr_lag_3",
    "Count_above_mean", "Count_below_mean", "Doy_of_Maximum_dates", "Doy_of_Minimum_dates",
    "Kurtosis", "Large_Standard_Deviation", "Maximum", "Mean", "Mean_abs_Change",
    "Mean_Change", "Mean_Second_Derivative_Central", "Median", "Minimum",
    "Quantile_pointzerofive", "Quantile_pointninefive", "Ratio_Beyond_Sigma_R1",
    "Ratio_Beyond_Sigma_R2", "Skewness", "Standard_Deviation", "Sum", "Symmetry_Looking",
    "TS_Complexity_Cid_ce", "Variance", "Variance_Larger_than_standard_deviation"
]

#%%
print(df8.head(10))
num_rows, num_columns = df8.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df8['Type'] = 'B6'
print(df8.head(10))

#%% File 9 B11
file_9 = "B11_34S_19E_258N.parquet"
df9 = pd.read_parquet(file_9)
pd.set_option("display.max_columns", None)
print(df9.head(10))

#%%
num_rows, num_columns = df9.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df9.columns = [
    "id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name",
    "ABS_Energy", "Absolute_Sum_of_Changes", "Autocorr_lag_1", "Autocorr_lag_2", "Autocorr_lag_3",
    "Count_above_mean", "Count_below_mean", "Doy_of_Maximum_dates", "Doy_of_Minimum_dates",
    "Kurtosis", "Large_Standard_Deviation", "Maximum", "Mean", "Mean_abs_Change",
    "Mean_Change", "Mean_Second_Derivative_Central", "Median", "Minimum",
    "Quantile_pointzerofive", "Quantile_pointninefive", "Ratio_Beyond_Sigma_R1",
    "Ratio_Beyond_Sigma_R2", "Skewness", "Standard_Deviation", "Sum", "Symmetry_Looking",
    "TS_Complexity_Cid_ce", "Variance", "Variance_Larger_than_standard_deviation"
]

#%%
print(df9.head(10))
num_rows, num_columns = df9.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df9['Type'] = 'B11'
print(df9.head(10))

#%% File 10 B11
file_10 = "B11_34S_19E_259N.parquet"
df10 = pd.read_parquet(file_10)
pd.set_option("display.max_columns", None)
print(df10.head(10))

#%%
num_rows, num_columns = df10.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df10.columns = [
    "id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name",
    "ABS_Energy", "Absolute_Sum_of_Changes", "Autocorr_lag_1", "Autocorr_lag_2", "Autocorr_lag_3",
    "Count_above_mean", "Count_below_mean", "Doy_of_Maximum_dates", "Doy_of_Minimum_dates",
    "Kurtosis", "Large_Standard_Deviation", "Maximum", "Mean", "Mean_abs_Change",
    "Mean_Change", "Mean_Second_Derivative_Central", "Median", "Minimum",
    "Quantile_pointzerofive", "Quantile_pointninefive", "Ratio_Beyond_Sigma_R1",
    "Ratio_Beyond_Sigma_R2", "Skewness", "Standard_Deviation", "Sum", "Symmetry_Looking",
    "TS_Complexity_Cid_ce", "Variance", "Variance_Larger_than_standard_deviation"
]

#%%
print(df10.head(10))
num_rows, num_columns = df10.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df10['Type'] = 'B11'
print(df10.head(10))

#%% File 11 B12
file_11 = "B12_34S_19E_258N.parquet"
df11 = pd.read_parquet(file_11)
pd.set_option("display.max_columns", None)
print(df11.head(10))

#%%
num_rows, num_columns = df11.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df11.columns = [
    "id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name",
    "ABS_Energy", "Absolute_Sum_of_Changes", "Autocorr_lag_1", "Autocorr_lag_2", "Autocorr_lag_3",
    "Count_above_mean", "Count_below_mean", "Doy_of_Maximum_dates", "Doy_of_Minimum_dates",
    "Kurtosis", "Large_Standard_Deviation", "Maximum", "Mean", "Mean_abs_Change",
    "Mean_Change", "Mean_Second_Derivative_Central", "Median", "Minimum",
    "Quantile_pointzerofive", "Quantile_pointninefive", "Ratio_Beyond_Sigma_R1",
    "Ratio_Beyond_Sigma_R2", "Skewness", "Standard_Deviation", "Sum", "Symmetry_Looking",
    "TS_Complexity_Cid_ce", "Variance", "Variance_Larger_than_standard_deviation"
]

#%%
print(df11.head(10))
num_rows, num_columns = df11.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df11['Type'] = 'B12'
print(df11.head(10))

#%% File 12 B12
file_12 = "B12_34S_19E_259N.parquet"
df12 = pd.read_parquet(file_12)
pd.set_option("display.max_columns", None)
print(df12.head(10))

#%%
num_rows, num_columns = df12.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df12.columns = [
    "id", "point", "fid", "SHAPE_AREA", "SHAPE_LEN", "crop_id", "crop_name",
    "ABS_Energy", "Absolute_Sum_of_Changes", "Autocorr_lag_1", "Autocorr_lag_2", "Autocorr_lag_3",
    "Count_above_mean", "Count_below_mean", "Doy_of_Maximum_dates", "Doy_of_Minimum_dates",
    "Kurtosis", "Large_Standard_Deviation", "Maximum", "Mean", "Mean_abs_Change",
    "Mean_Change", "Mean_Second_Derivative_Central", "Median", "Minimum",
    "Quantile_pointzerofive", "Quantile_pointninefive", "Ratio_Beyond_Sigma_R1",
    "Ratio_Beyond_Sigma_R2", "Skewness", "Standard_Deviation", "Sum", "Symmetry_Looking",
    "TS_Complexity_Cid_ce", "Variance", "Variance_Larger_than_standard_deviation"
]

#%%
print(df12.head(10))
num_rows, num_columns = df12.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%%
df12['Type'] = 'B12'
print(df12.head(10))

#%% Merge Files
dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12]
df_merged = pd.concat(dfs, ignore_index=True)
print(df_merged.head(5))

#%%
num_rows, num_columns = df_merged.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

#%% Check for missing values
print(df_merged.isnull().sum())

#%% Check unique values in categorical columns
print(df_merged[['Type', 'crop_name']].nunique())

#%%Distribution of Numerical Features
import matplotlib.pyplot as plt
import seaborn as sns

df_merged.hist(figsize=(15, 12), bins=30)
plt.tight_layout()
plt.show()

#%%Distribution of Type (Band Categories)
sns.countplot(data=df_merged, x='Type')
plt.xticks(rotation=45)
plt.title("Distribution of Type Categories")
plt.show()

#%%Correlation Between Features
plt.figure(figsize=(12, 8))
sns.heatmap(df_merged.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

#%%Crop-Type Analysis
df_merged['crop_name'].value_counts().plot(kind='bar', figsize=(12, 6))
plt.title("Crop Name Distribution")
plt.xlabel("Crop Name")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

#%% Field-Level Analysis (fid)
print(f"Number of unique fields: {df_merged['fid'].nunique()}")
df_merged.groupby('fid').mean().drop(columns=['id', 'point', 'crop_id']).head()

#%% Outlier Detection Using Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_merged[['Mean', 'Maximum', 'Minimum', 'Variance', 'Standard_Deviation']])
plt.title("Boxplot of Key Numerical Features")
plt.xticks(rotation=45)
plt.show()











