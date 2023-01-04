import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

data_path = "data/processed/sample_points.csv"
save_image_path = "images/{}_{}.svg"

def main():
    # Load data
    data = pd.read_csv(data_path, index_col=0)

    # Boxplots
    for title, var in variables.items():
        fig = group_boxplot(data, var, "species", title, xticklabels[title], False)
        fig.savefig(save_image_path.format("boxplot", title.lower()))

    # Get X and y
    X = data[data.columns[:-5]].copy()
    y = data.vegetation.copy()
    s = data.species.copy()

    # Scale X data
    scaler = MinMaxScaler()
    X.iloc[:,:] = scaler.fit_transform(X)

    # PCA scatter plot
    fig = plot_variable_importance(X, y, s)
    fig.savefig(save_image_path.format("pca", "train"))

    return None

if __name__ == "__main__":
    main()
