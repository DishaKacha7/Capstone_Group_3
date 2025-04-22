import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box

def main():
    # -----------------------------------------------------
    # 1) PARAMETERS
    # -----------------------------------------------------
    geojson_path = "/home/ubuntu/sai/Capstone_Group_3/src/updated_combined_fields.geojson"
    patch_size = 100.0  # in map units
    patches_geojson_out = "/home/ubuntu/sai/Capstone_Group_3/src/Deep Learning/updated_patch_level.geojson"

    # -----------------------------------------------------
    # 2) LOAD THE FIELDS & PLOT
    # -----------------------------------------------------
    gdf = gpd.read_file(geojson_path)
    print("Original fields GeoDataFrame:")
    print(gdf.tail())
    print(gdf.columns)
    print("CRS:", gdf.crs)
    print("Number of fields:", gdf['fid'].nunique())
    print("Number of Crops:", gdf['crop_name'].value_counts().sum())

    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.boundary.plot(ax=ax, color='black', linewidth=1)
    ax.set_title("Original Field Boundaries")
    plt.show()

    # -----------------------------------------------------
    # 3) CREATE PATCHES PER FIELD
    # -----------------------------------------------------
    patches_list = []
    # For tracking field ids that have at least one patch
    fields_with_patches = set()

    for idx, field in gdf.iterrows():
        field_id = field['fid']
        crop_name = field["crop_name"]  # or adjust if needed
        field_geom = field.geometry

        # Field bounding box and dimensions
        minx, miny, maxx, maxy = field_geom.bounds
        width = maxx - minx
        height = maxy - miny
        area = width * height

        if area < patch_size * patch_size:
            # If field is smaller than patch size, add as one patch.
            patches_list.append({
                "field_id": field_id,
                "crop_name": crop_name,
                "geometry": field_geom
            })
            fields_with_patches.add(field_id)
        else:
            # Create a grid of top-left corners with step = patch_size
            x_coords = np.arange(minx, maxx, patch_size)
            y_coords = np.arange(miny, maxy, patch_size)
            added_patch = False  # track if at least one patch is added for this field
            for x in x_coords:
                for y in y_coords:
                    patch_poly = box(x, y, x + patch_size, y + patch_size)
                    # Only add patches fully within the field
                    if patch_poly.within(field_geom):
                        patches_list.append({
                            "field_id": field_id,
                            "crop_name": crop_name,
                            "geometry": patch_poly
                        })
                        added_patch = True
            if added_patch:
                fields_with_patches.add(field_id)

    # After processing all fields, check for missing field ids.
    all_field_ids = set(gdf['fid'].unique())
    missing_field_ids = all_field_ids - fields_with_patches
    print(f"Missing field ids (no patches created): {missing_field_ids}")

    # For each missing field, add the entire field as one patch.
    for fid in missing_field_ids:
        field_row = gdf[gdf['fid'] == fid].iloc[0]
        patches_list.append({
            "field_id": fid,
            "crop_name": field_row["crop_name"],
            "geometry": field_row.geometry
        })

    # Create a GeoDataFrame of patches
    patches_gdf = gpd.GeoDataFrame(patches_list, crs=gdf.crs)
    print(f"\nCreated {len(patches_gdf)} patches (including full-field patches for missing fields).")

    # -----------------------------------------------------
    # 4) PLOT THE PATCHES OVER THE FIELDS
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    # gdf.boundary.plot(ax=ax, color='black', linewidth=1)
    patches_gdf.plot(ax=ax, color='red', alpha=0.5, edgecolor='red')
    ax.set_title("Field Boundaries with Patch Grid")
    plt.show()

    # -----------------------------------------------------
    # 5) SAVE THE PATCHES
    # -----------------------------------------------------
    if patches_geojson_out:
        patches_gdf.to_file(patches_geojson_out, driver='GeoJSON')
        print(f"Saved patch GeoDataFrame to {patches_geojson_out}")

    print(patches_gdf['crop_name'].value_counts())
    print("Unique field_id count:", patches_gdf['field_id'].nunique())

if __name__ == "__main__":
    main()
