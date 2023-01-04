import geopandas as gpd
import rasterio

from functions.gis_utils import sample_points

sample_path = "data/processed/sample_points.csv"
points_path = "data/external/control_points/control_points.shp"
raster_path = "data/raster/train_test_image/train_img.tiff"

def main():
    points = gpd.read_file(points_path)
    
    with rasterio.open(raster_path, "r") as raster:
        sample = sample_points(points, raster)
        bands = raster.descriptions

    sample = sample.dropna(subset=bands)

    sample.to_csv(sample_path)

    return None

if __name__ == "__main__":
    main()