import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray

from sklearn.preprocessing import MinMaxScaler

from functions.gis_utils import plot_samples
from functions.stats_utils import group_boxplot, plot_variable_importance

plt.style.use("src/style.mplstyle")

variables = {
    "Reflectance": ["BLUE", "GREEN", "RED", "NIR", "SWIR"],
    "Temperature [K]": ["TEMPERATURE"],
    "Distance [m]": ["DISTANCE_DRAINAGES", "DISTANCE_SHORELINE"],
    "Elevation [m]": ["ELEVATION"],
    "Slope [°]": ["SLOPE"],
    "Spectral Indices": ["NDVI", "EVI", "NDMI", "SAVI", "GCI", "BGR"]
}

xticklabels = {
    "Reflectance": ["Blue", "Green", "Red", "NIR", "SWIR"],
    "Temperature [K]": ["Temperature"],
    "Distance [m]": ["D. to drainages", "D. to shoreline"],
    "Elevation [m]": ["Elevation"],
    "Slope [°]": ["Slope"],
    "Spectral Indices": ["NDVI", "EVI", "NDMI", "SAVI", "GCI", "BGR"]
}

data_path = "data/processed/sample_points.geojson"
raster_path = "data/raster/train_test_image/train_img.tiff"
save_image_path = "images/{:02d}_{}.svg"

def main():
    # Load data
    data = gpd.read_file(data_path, index_col=0)

    # Load raster
    img = rioxarray.open_rasterio(raster_path, decode_cords="All")

    # Map
    fig1 = plt.figure(figsize=(6, 6))
    ax = fig1.add_subplot(111)
    plot_samples(
        data, "species", img, 
        bands=[3, 2, 1], 
        zlims=(0.0, 0.3), 
        ylims=(1.306e6, 1.346e6), 
        xlims=(865000, 915000), 
        ax=ax
    )

    fig1.savefig(save_image_path.format(1, "map"))

    # Boxplots
    fig2, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 10))
    axs = axs.reshape(-1)

    for ax, (title, var) in zip(axs, variables.items()):
        group_boxplot(
            data, var, "species", title, xticklabels[title], False, ax
        )
        
    axs[3].legend(title="species")

    fig2.savefig(save_image_path.format(2, "boxplots"))

    # Get X and y
    X = data[data.columns[:-5]].copy()
    y = data.vegetation.copy()
    s = data.species.copy()

    # Scale X data
    scaler = MinMaxScaler()
    X.iloc[:,:] = scaler.fit_transform(X)

    # PCA scatter plot
    fig3 = plt.figure()
    ax = fig3.add_subplot()
    plot_variable_importance(X, y, s, ax)
    
    fig3.savefig(save_image_path.format(3, "scatter_pca"))

    plt.show()

    return None

if __name__ == "__main__":
    main()
