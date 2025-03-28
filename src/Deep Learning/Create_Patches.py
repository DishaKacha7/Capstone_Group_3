#%%
## -- Description --
## Code to create square patches from Geojson file of combined fields
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box


def main():
    # -----------------------------------------------------
    # 1) Parameters
    # -----------------------------------------------------
    # Path hard coded for now, to be updated

    geojson_path = "/home/ubuntu/sai/Capstone_Group_3/src/combined_fields.geojson"
    patch_size = 100.0  # size is set in map units
    
    # Path hard coded for now, to be updated
    patches_geojson_out = "/home/ubuntu/sai/Capstone_Group_3/src/Deep Learning/patch_level.geojson"

    # -----------------------------------------------------
    # 2) Loading the fields and plotting the polygons
    # -----------------------------------------------------
    gdf = gpd.read_file(geojson_path)
    print("Original fields GeoDataFrame:")
    print(gdf.head())
    print("CRS:", gdf.crs)
    print("Number of fields:", len(gdf))

    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.boundary.plot(ax=ax, color='black', linewidth=1)
    ax.set_title("Original Field Boundaries")
    plt.show()

    # -----------------------------------------------------
    # 3) Iterating over each field polygon and assigning patches
    # -----------------------------------------------------
    patches_list = []

    for idx, field in gdf.iterrows():
        field_id = idx + 1
        crop_name = field["crop_name"]
        field_geom = field.geometry

        # Field bounding box
        minx, miny, maxx, maxy = field_geom.bounds

        # Create a grid of top-left corners
        # We step from minx to maxx in increments of patch_size
        # and from miny to maxy in increments of patch_size
        x_coords = np.arange(minx, maxx, patch_size)
        y_coords = np.arange(miny, maxy, patch_size)

        for x in x_coords:
            for y in y_coords:
                # Creates a patch as a square
                patch_poly = box(x, y, x + patch_size, y + patch_size)

                if patch_poly.within(field_geom):
                    patches_list.append({
                        "field_id": field_id,
                        "crop_name": crop_name,
                        "geometry": patch_poly
                    })

    # Create a GeoDataFrame of all the patches created
    patches_gdf = gpd.GeoDataFrame(patches_list, crs=gdf.crs)
    print(f"\nCreated {len(patches_gdf)} patches (fully inside their fields).")

    # -----------------------------------------------------
    # 4) Plotting the patches made, to verify the validity
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
 
    # gdf.boundary.plot(ax=ax, color='black', linewidth=1)
    patches_gdf.plot(ax=ax, color='red', alpha=0.5, edgecolor='red')
    ax.set_title("Field Boundaries with Patch Grid")
    plt.show()

    # -----------------------------------------------------
    # 5) Saving the patches
    # -----------------------------------------------------
    if patches_geojson_out:
        patches_gdf.to_file(patches_geojson_out, driver='GeoJSON')
        print(f"Saved patch GeoDataFrame to {patches_geojson_out}")

if __name__ == "__main__":
    main()
#%%
patches_gdf=gpd.read_file('patch_level.geojson')
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
num_to_sample = 200000
if len(patches_gdf) > num_to_sample:
    subset = patches_gdf.sample(n=num_to_sample, random_state=42)
else:
    subset = patches_gdf

fig, ax = plt.subplots(figsize=(10, 8))
subset.plot(ax=ax, color='red', alpha=0.7, edgecolor='white')
ax.set_title(f"Random Subset of {num_to_sample} Patches")
plt.show()

#%%
