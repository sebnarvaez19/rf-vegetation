import geopandas as gpd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

from functions.model import rf_classifier

plt.style.use("src/style.mplstyle")

# Define paths
raster_path = "data/raster/train_test_image/train_img.tiff"
roi_path = "data/shapefile/roi/roi.shp"
save_path = "data/raster/train_test_image/classification.tiff"
points_path = "data/processed/sample_points.geojson"
save_image_path = "images/{:02d}_{}.png"

def main():
    # Load image and roi
    img = rioxarray.open_rasterio(raster_path, decode_coords="all")
    roi = gpd.read_file(roi_path).geometry

    # Clip image
    img = img.rio.clip(roi, all_touched=False)

    # Filter bands of interest
    bands = [2, 4, 5, 6, 7, 13, 15, 16]
    img = img.sel(band=bands)

    # Convert to array 
    data = img.to_numpy()
    x = img.coords["x"].to_numpy()
    y = img.coords["y"].to_numpy()

    dat2 = []

    for i in range(data.shape[0]):
        dat2.append(data[i,:,:].reshape(-1))

    dat2 = np.array(dat2).T

    # Find the valid vaues
    valids = np.isfinite(dat2).all(axis=1)

    # Predict
    pred = rf_classifier.predict(dat2)

    # Restore the original dimension of the raster
    imgc = np.full(valids.shape, np.nan)
    imgc[valids] = pred

    imgc = imgc.reshape(data.shape[1:])

    # Define the CRS and transformation
    crs = CRS.from_epsg(32618)
    transform = Affine.translation(x[0]-15, y[-1]-15) * Affine.scale(30, 30)

    # Save raster
    with rasterio.open(
        save_path,
        mode="w",
        driver="GTiff",
        height=img.coords["y"].to_numpy().shape[0]-1,
        width=img.coords["x"].to_numpy().shape[0],
        count=1,
        dtype=imgc.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(np.flipud(imgc), 1)

    # Plot raster
    gdf = gpd.read_file(points_path, index_col=0)
    
    species = gdf.sort_values(by="vegetation").species.unique()
    cmap = ListedColormap([
        "#0000ff", "#689d4d", "#ffff9d", "#ad256b", "#c4ff4d", "#b88454"
    ])
    
    handles = [Patch(color=c, label=l) for c, l in zip(cmap.colors, species)]
    
    imgc = rioxarray.open_rasterio(save_path, decode_coords="all")
    
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    imgc.plot(cmap=cmap, ax=ax, add_colorbar=False)

    ax.legend(
        handles=handles, loc=2, title="species", frameon=True, framealpha=0.5
    )
    ax.set_title(None)

    fig.savefig(save_image_path.format(7, "train_image_classified"))

    plt.show()

if __name__ == "__main__":
    main()