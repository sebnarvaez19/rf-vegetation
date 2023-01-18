import numpy as np
import rioxarray
import xarray

from functions.model import rf_classifier

# Define paths
img_path = "data/raster/annual_rasters/{}_clipped.tif"
save_path = "data/processed/annual_vegetation_coverage.tif"

# Codes for classifications
codes = ["Water", "Woodland", "Dessert", "Mangrove", "Thorn", "Shrubs"]


def main():
    data = []
    t = []

    for year in range(2000, 2023):
        # Read and classify raster
        img = rioxarray.open_rasterio(img_path.format(year), decode_coords="all")
        img = img.to_dataframe(name=f"{year}").reset_index().drop("spatial_ref", axis=1)
        img = img.pivot(index=["x", "y"], columns="band", values=f"{year}")
        img["valids"] = img.notna().all(axis=1)
        img["classified"] = np.nan

        img.classified[img.valids == True] = rf_classifier.predict(img.iloc[:, :-2])
        img = img.reset_index()

        # Get the dimensions of the rasters
        t.append(year)
        x = img.x.unique()
        y = img.y.unique()

        # Save classified rasters
        z = img.classified.values.reshape(x.shape[0], y.shape[0]).T

        data.append(z)

    data = np.array(data)
    t = np.array(t)

    # Create raset
    data = xarray.DataArray(
        data=data,
        dims=("time", "y", "x"),
        coords={"time": t, "y": y, "x": x},
        attrs={"codes": codes},
    )

    data = data.rio.write_crs("EPSG:32618")
    data = data.rio.set_spatial_dims(x_dim="x", y_dim="y")

    data.rio.to_raster(save_path)

    return None


if __name__ == "__main__":
    main()
