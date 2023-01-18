import ee
import geemap
import geopandas as gpd
import rasterio
from rasterio.merge import merge
import rioxarray
import xarray
from statgis.landsat_functions import landsat_cloud_mask, landsat_scaler

from functions.gee_processing import add_dem, add_distance, add_indices, rename_bands

ee.Initialize()

variables = [
    "GREEN",
    "NIR",
    "SWIR",
    "TEMPERATURE",
    "NDVI",
    "ELEVATION",
    "DISTANCE_DRAINAGES",
    "DISTANCE_SHORELINE",
]

margin = 0.04

roi_path = "data/shapefile/roi/roi.shp"
img_path = "data/raster/annual_rasters/"


def main():
    # Load region of interest and set the limits for the images
    gdf = gpd.read_file(roi_path).to_crs(4326)
    lims = [
        gdf.bounds.minx.min() - margin,
        gdf.bounds.miny.min() - margin,
        gdf.bounds.maxx.max() + margin,
        gdf.bounds.maxy.max() + margin,
    ]

    bbox = ee.Geometry.BBox(*lims)
    fnet = geemap.fishnet(bbox, rows=2, cols=2)

    # Load the shoreline and the drainages to calculate their distance rasters
    shoreline = ee.FeatureCollection("projects/ee-rf-vegetation-guajira/assets/shoreline")
    drainages = ee.FeatureCollection("projects/ee-rf-vegetation-guajira/assets/drainage")

    L7 = (
        ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
          .filterBounds(bbox)
          .map(landsat_cloud_mask)
          .map(landsat_scaler)
          .map(rename_bands)
          .map(add_indices)
    )

    for year in range(2000, 2023):
        # Reduce image to mean
        img = L7.filter(ee.Filter.calendarRange(year, year, "year")).mean()

        # Add dem and distance to shoreline and drainages
        img = add_dem(img)
        img = add_distance(img, drainages, "DISTANCE_DRAINAGES")
        img = add_distance(img, shoreline, "DISTANCE_SHORELINE")

        # Select the bands of interest
        img = img.select(variables)

        # Define the image save path
        save_img_path = img_path + f"{year}/"

        # Save images in tiles
        geemap.download_ee_image_tiles(
            img, fnet, save_img_path, scale=30, unmask_value=None, crs="EPSG:32618"
        )

        # Open and merge tiles
        rasters = [rasterio.open(save_img_path + f"{i}.tif") for i in range(1, 10)]
        mosaic, transform = merge(rasters)

        # Update dimensions
        meta_data = rasters[0].meta.copy()
        meta_data.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
            }
        )

        # Close tiles
        rasters = [raster.close() for raster in rasters]

        # Save merged raster
        with rasterio.open(img_path + f"{year}.tif", "w", **meta_data) as src:
            src.write(mosaic)
            src.descriptions = tuple(variables)

        # Clip with rioxarray
        roi = gdf.to_crs(32618).geometry

        img = rioxarray.open_rasterio(img_path + f"{year}.tif", decode_coords="all")
        img = img.rio.clip(roi, all_touched=False)

        # Save final image
        img.rio.to_raster(img_path + f"{year}_clipped.tif")

    return None


if __name__ == "__main__":
    main()
