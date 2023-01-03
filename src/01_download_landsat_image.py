# Imports
import os

import ee
import geemap
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from statgis.landsat_functions import landsat_cloud_mask, landsat_scaler

from functions.gee_processing import add_dem, add_distance, add_indices, add_land_cover, rename_bands

# initialize Earth Engine
ee.Initialize()

# Define paths to load and save data
poc_path = "data/external/randomfor_controlpoints_utm.shp"
save_image = "data/raster/train_test_image"
save_parts = save_image + "/parts/"

# Add a margin to the images
margin = 0.04


def main():
    # Load points data
    poc = geemap.shp_to_ee(poc_path)
    gdf = gpd.read_file(poc_path).to_crs(epsg=4326)

    # extract from the points the images bounds
    lims = [
        gdf.bounds.minx.min() - margin,
        gdf.bounds.miny.min() - margin,
        gdf.bounds.maxx.max() + margin,
        gdf.bounds.maxy.max() + margin,
    ]

    # Create a rectangle from the bounds and the nest of regions to save
    # the image
    bbox = ee.Geometry.BBox(*lims)
    fnet = geemap.fishnet(bbox, rows=2, cols=2)

    # Load the shoreline and the drainages to calculate their distance rasters
    shoreline = ee.FeatureCollection("projects/ee-rf-vegetation-guajira/assets/shoreline")
    drainages = ee.FeatureCollection("projects/ee-rf-vegetation-guajira/assets/drainage")

    # Load landsat image collection and filter by date and region, also scale
    # the images and mask the clouds and calculate the spatial indeces related
    # to vegetation
    L7 = (
        ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
        .filterBounds(poc)
        .filterDate("2007-02-01", "2007-03-31")
        .map(landsat_cloud_mask)
        .map(landsat_scaler)
        .map(rename_bands)
        .map(add_indices)
    )

    # Reduce the images to the mean
    img = L7.mean()

    # Add the elevation and slope
    img = add_dem(img, slope=True)

    # Add the distance to drainage and shoreline
    img = add_distance(img, drainages, "DISTANCE_DRAINAGES")
    img = add_distance(img, shoreline, "DISTANCE_SHORELINE")

    # Add land cover classification
    img = add_land_cover(img)

    # Download the image by tiles because the large image size
    geemap.download_ee_image_tiles(
        img, fnet, save_parts, scale=30, unmask_value=None, crs="EPSG:32618"
    )

    # Load downloaded raster
    paths_rasters = [save_parts + file for file in os.listdir(save_parts)]
    rasters = [rasterio.open(raster) for raster in paths_rasters]

    # Merge the raster in a mosaic
    mosaic, transform = merge(rasters)

    # Get the metadata of on image and update the width, height and tranform
    # to save the mosaic
    meta_data = rasters[0].meta.copy()
    meta_data.update(
        {
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
        }
    )

    # Save the mosaic
    with rasterio.open(save_image + "/landsat_img.tiff", "w", **meta_data) as src:
        src.write(mosaic)
        src.descriptions = tuple(img.bandNames().getInfo())
        
    return None


if __name__ == "__main__":
    main()
