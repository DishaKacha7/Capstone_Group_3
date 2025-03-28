#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

# ---------------------------------------------------------------------
# 1) PARAMETERS & FILE PATHS
# ---------------------------------------------------------------------
# The directory containing 6 subfolders (one per band), each with .tif files
BOX_DIR = "/home/ubuntu/box/SA_raw_imagery"

# The patch-level GeoJSON where each feature = one patch polygon
PATCH_GEOJSON = "patch_level.geojson"

# Final output Parquet
OUTPUT_PARQUET = "/home/ubuntu/sai/Capstone_Group_3/src/Data/patch_level_data.parquet"


# ---------------------------------------------------------------------
# 2) FUNCTION TO RECURSIVELY FIND ALL .TIF UNDER BOX_DIR
# ---------------------------------------------------------------------
def find_raster_files(root_dir):
    """
    Recursively search 'root_dir' for all .tif files, build a dictionary:
        { "filename_without_extension": "/full/path/to/file.tif", ... }
    If the same filename appears more than once, the last one found
    overwrites the previous in the dictionary. (You can handle duplicates if needed.)
    """
    raster_dict = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(".tif"):
                full_path = os.path.join(dirpath, f)
                key = os.path.splitext(f)[0]  # e.g. "SA_B2_1C_2017_01"
                if key in raster_dict:
                    print(f"Warning: Duplicate key '{key}' found. Overwriting previous entry.")
                raster_dict[key] = full_path
    return raster_dict


# ---------------------------------------------------------------------
# 3) COLLATE PATCH DATA
# ---------------------------------------------------------------------
def collate_patch_data(patch_geojson, raster_files, output_parquet):
    """
    Loads each patch polygon from patch_geojson, masks each raster,
    flattens pixel arrays, and saves a DataFrame to output_parquet.
    The resulting columns:
      - patch_id, field_id, crop_name, row, col, plus one column per band key.
    """
    print("Loading patch-level GeoJSON...")
    gdf = gpd.read_file(patch_geojson)
    print("GeoDataFrame shape:", gdf.shape)
    print("Columns:", gdf.columns.tolist())
    print("CRS:", gdf.crs)
    print("Bounds:", gdf.total_bounds)

    # If we have no rasters, there's nothing to do
    if not raster_files:
        print("No .tif files were found in the BOX_DIR. Exiting...")
        return pd.DataFrame()

    # Reproject patches to match the first raster's CRS
    # We'll pick the first item in raster_files as the reference
    ref_raster = next(iter(raster_files.values()))
    with rasterio.open(ref_raster) as src_ref:
        raster_crs = src_ref.crs

    if gdf.crs != raster_crs:
        print(f"Reprojecting patches from {gdf.crs} to {raster_crs}...")
        gdf = gdf.to_crs(raster_crs)

    rows_list = []

    print("\nMasking each patch. This may take a while if many patches...")
    # Loop over each patch feature
    for idx, patch_row in gdf.iterrows():
        # If you already have a 'patch_id' in the GeoJSON, use patch_row["patch_id"].
        patch_id = idx + 1
        field_id = patch_row.get("field_id", None)
        crop_name = patch_row.get("crop_name", None)

        geom = [mapping(patch_row.geometry)]

        band_arrays = {}
        # For each raster
        for band_key, tif_path in raster_files.items():
            # Read & mask
            with rasterio.open(tif_path) as src:
                out_image, out_transform = mask(src, geom, crop=True)
                # out_image shape => (1, height, width) if single band
                band_arrays[band_key] = out_image[0]

        # Flatten pixel arrays
        sample_shape = next(iter(band_arrays.values())).shape  # (H, W)
        H, W = sample_shape

        flattened_bands = {bk: band_arrays[bk].flatten() for bk in band_arrays}

        # Build pixel-level records
        for r in range(H):
            for c in range(W):
                flat_idx = r * W + c
                row_data = {
                    "patch_id": patch_id,
                    "field_id": field_id,
                    "crop_name": crop_name,
                    "row": r,
                    "col": c
                }
                for bk in flattened_bands:
                    row_data[bk] = flattened_bands[bk][flat_idx]

                rows_list.append(row_data)

    df = pd.DataFrame(rows_list)
    print(f"\nFinal patch-level DataFrame shape: {df.shape}")
    print("Columns in DataFrame:", df.columns.tolist())
    print("\nHead:\n", df.head(5))

    # Basic stats
    print("\nValue counts of patch_id (top 10):")
    print(df["patch_id"].value_counts().head(10))
    print("\nValue counts of field_id (top 10):")
    print(df["field_id"].value_counts(dropna=False).head(10))
    print("\nValue counts of crop_name (top 10):")
    print(df["crop_name"].value_counts(dropna=False).head(10))

    # Save to Parquet
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"\nSaved patch-level data to: {output_parquet}")

    return df

# ---------------------------------------------------------------------
# 4) MAIN
# ---------------------------------------------------------------------
def main():
    # 1) Recursively find all .tif under BOX_DIR
    raster_files = find_raster_files(BOX_DIR)
    print(f"Found {len(raster_files)} .tif files in '{BOX_DIR}'.")

    # 2) Collate data for each patch
    collate_patch_data(
        patch_geojson=PATCH_GEOJSON,
        raster_files=raster_files,
        output_parquet=OUTPUT_PARQUET
    )

if __name__ == "__main__":
    main()
