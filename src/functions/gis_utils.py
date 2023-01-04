import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray

from matplotlib.axes import Axes
from typing import Sequence


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
    geom = gdf.geometry
    labels = raster.descriptions
    coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    sample = np.array([s for s in raster.sample(coords)])

    sample = pd.DataFrame(sample, columns=labels)
    sample = pd.concat([sample, gdf], axis=1)

    sample = gpd.GeoDataFrame(sample[sample.columns[:-1]], geometry=geom)

    return sample


def plot_samples(
    samples: gpd.GeoDataFrame,
    variable: str,
    raster: xarray.Dataset,
    bands: list[int] | int = 1,
    zlims: Sequence[float] | None = None,
    ylims: Sequence[float] | None = None,
    xlims: Sequence[float] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Plotthe sample in a raster opened with rioxarray.

    Parameters
    ----------
    data : geopandas.GeoDataFrame
        Data frame with the samples data

    variable : str
        Variable of interest

    raster : xarray.Dataset
        Raster.

    bands : Sequence[int] | int (optional) = 1
        Raster bands to plot.

    zlims : Sequence[float] | None (optional) = None
        Limits of raster values.

    ylims : Sequence[float] | None (optional) = None
        Y-Axis limits.

    xlims : Sequence[float] | None (optional) = None
        X-Axis limits.

    ax : matplotlib.axes.Axes | None (optional) = None
        Axes to draw the plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Map.
    """
    
    # Squeez bands
    raster = raster.sel(band=bands)
    raster = raster.squeeze()

    # If there not axes, create one
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # If there are not zlims, plot raster without them
    if zlims == None:
        raster.plot.imshow(ax=ax)
    else:
        raster.plot.imshow(vmin=zlims[0], vmax=zlims[1], ax=ax)

    # Add the samples
    for sample in samples[variable].unique():
        samples[samples[variable] == sample].plot(label=sample, ax=ax, alpha=0.5)
    
    # Add the legend
    ax.legend(title=variable, loc=0, frameon=True, framealpha=0.5)

    # Remove default title
    ax.set_title(None)

    if ylims != None:
        ax.set_ylim(ylims[0], ylims[1])

    if xlims != None:
        ax.set_xlim(xlims[0], xlims[1])
        
    return ax