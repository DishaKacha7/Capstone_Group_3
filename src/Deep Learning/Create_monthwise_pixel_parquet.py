"""
Generates a field-level pixel DataFrame from field-boundary GeoJSON and multi-temporal rasters.

For each field polygon, this script:
  1. Reads a GeoJSON of field boundaries with columns ['fid', 'SHAPE_AREA', 'SHAPE_LEN', 'geometry'].
  2. Finds all .tif raster files under BOX_DIR (e.g., month-wise bands).
  3. Masks each raster to the field polygon, flattens pixel values.
  4. Emits rows with: point_id, fid, SHAPE_AREA, SHAPE_LEN, <band_key> for each band.
  5. Writes the combined DataFrame to OUTPUT_PARQUET.

"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

BOX_DIR = "/home/ubuntu/box/SA_raw_imagery"               # This is the box folder loaded in the instance
FIELD_GEOJSON = "X_testing_34S_20E_259N.geojson"
OUTPUT_PARQUET = "field_pixel_data.parquet"
CHUNK_SIZE = 200000

def find_raster_files(root_dir):
    raster_dict = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".tif"):
                key = os.path.splitext(fname)[0]  # e.g. "SA_B2_1C_2017_01"
                path = os.path.join(dirpath, fname)
                raster_dict[key] = path
    return raster_dict

def extract_field_pixels(geojson_file, raster_files, output_parquet, chunk_size=CHUNK_SIZE):
    # Load field boundaries
    gdf = gpd.read_file(geojson_file)
    print(f"Loaded {len(gdf)} fields from '{geojson_file}'.")

    ref_tif = next(iter(raster_files.values()))
    with rasterio.open(ref_tif) as src0:
        raster_crs = src0.crs
    if gdf.crs != raster_crs:
        print(f"Reprojecting fields from {gdf.crs} to {raster_crs}...")
        gdf = gdf.to_crs(raster_crs)

    rows = []
    chunks = []
    point_id = 1

    # Looping over fields
    for idx, row in gdf.iterrows():
        fid = row['fid']
        area = row['SHAPE_AREA']
        length = row['SHAPE_LEN']
        geom = [mapping(row.geometry)]

        # Masking each raster, then flattening
        band_data = {}
        for key, path in raster_files.items():
            with rasterio.open(path) as src:
                arr, _ = mask(src, geom, crop=True)
            band_data[key] = arr[0].flatten()

        n_pix = len(next(iter(band_data.values())))

        for i in range(n_pix):
            rec = {
                'point_id': point_id,
                'fid': fid,
                'SHAPE_AREA': area,
                'SHAPE_LEN': length
            }
            for key, vals in band_data.items():
                rec[key] = vals[i]
            rows.append(rec)
            point_id += 1

            # flush chunk
            if len(rows) >= chunk_size:
                df_chunk = pd.DataFrame(rows)
                chunks.append(df_chunk)
                print(f"Chunk of {len(df_chunk)} rows appended.")
                rows = []
    if rows:
        df_chunk = pd.DataFrame(rows)
        chunks.append(df_chunk)
        print(f"Final chunk of {len(df_chunk)} rows appended.")

    df_all = pd.concat(chunks, ignore_index=True)
    print(f"Total pixels: {df_all.shape[0]}, cols: {df_all.shape[1]}")
    df_all.to_parquet(output_parquet, index=False)
    print(f"Saved field-pixel parquet to '{output_parquet}'")

# Main function
if __name__ == '__main__':
    rasters = find_raster_files(BOX_DIR)
    print(f"Found {len(rasters)} raster files under '{BOX_DIR}'.")
    extract_field_pixels(FIELD_GEOJSON, rasters, OUTPUT_PARQUET)
