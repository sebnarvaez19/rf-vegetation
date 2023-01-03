import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


def sample_points(
    gdf: gpd.GeoDataFrame, raster: rasterio.io.DatasetReader
) -> gpd.GeoDataFrame:
    """
    Sample the raster values in all bands and add to a points dataframe.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Points dataframe of interest to sample the raster.

    raster : rasterio.io.DataReader
        Raster to sample.

    Returns
    -------
    sample : geopandas.GeoDataFrame
        Geodataframe with the raster values
    """

    labels = raster.descriptions
    coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    sample = np.array([s for s in raster.sample(coords)])

    sample = pd.DataFrame(sample, columns=labels)
    sample = pd.concat([sample, gdf], axis=1)

    return sample
