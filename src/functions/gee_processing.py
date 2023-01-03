import ee


def rename_bands(image: ee.Image) -> ee.Image:
    """
    Function to extract the Landsat 5 and 7 bands of interest and rename it
    to coloquial names.
    Parameters
    ----------
    image : ee.Image
        Image of interest.

    Returns
    -------
    image : ee.Image
        Image with bands filtereds and renamed.
    """

    # Define the original and the new names.
    ori_names = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B7", "ST_B6"]
    new_names = ["BLUE", "GREEN", "RED", "NIR", "SWIR", "TEMPERATURE"]

    # Select the bands with the original names an rename them.
    image = image.select(ori_names).rename(new_names)

    return image


def add_indices(image: ee.Image) -> ee.Image:
    """
    Function to add the Normalized Difference Vegetation Index (NDVI),
    Enhanced Vegetation Index (EVI), Normalized Difference Moisture
    Index (NDMI), Soil Adjusted Vegetation Index (SAVI) and Green
    Coverage Index (GCI) to a image using the expression method.

    Parameters
    ----------
    image : ee.Image
        Image of interest.

    Returns
    -------
    image : ee.Image
        Image with the spatial indices.
    """

    # Define the indeces formulas.
    indices = {
        "NDVI": "(b('NIR') - b('RED'))/(b('NIR') + b('RED'))",
        "EVI": "2.5*((b('NIR') - b('RED'))/ \
                (b('NIR') + 6*b('RED') - 7.5*b('BLUE') + 1))",
        "NDMI": "(b('NIR') - b('RED'))/(b('NIR') + b('RED'))",
        "SAVI": "((b('NIR') - b('RED'))/(b('NIR') + b('RED') + 0.5))*(1.5)",
        "GCI": "(b('NIR')/b('GREEN')) - 1",
        "BGR": "b('BLUE')/b('GREEN')"
    }

    # For each index, calculate it and add it to the image.
    for name, formula in indices.items():
        image = image.addBands(
            image.expression(formula).toFloat().rename(name), None, True
        )

    return image


def add_dem(image: ee.Image, slope: bool = False, dem: str = "SRTM_30m") -> ee.Image:
    """
    Function to add a Digital Elevation Model (DEM) to an image.

    Paremeters
    ----------
    image : ee.Image
        Image of interest.

    slope : bool, optional (False)
        If True, calculate the slope in deegres and add it to the image.

    dem : str, optional ("SRTM_30m")
        key to select the dem to add:
        - "SRTM_30m": USGS SRTM DEM with gaps (30 m).
        - "SRTM_90m": USGS SRTM DEM with gaps filled (90 m).
        - "HydroSHED_15arc": WWF HydroSHED DEM (~470 m).
        - "GTOPO_30arc": USGS GTOPO DEM (~1 km).

    Returns
    -------
    image : ee.Image
        Image with the DEM band (and Slope if it was required).
    """

    # Select the DEM.
    match dem:
        case "SRTM_30m":
            dem_path = "USGS/SRTMGL1_003"
        case "SRTM_90m":
            dem_path = "CGIAR/SRTM90_V4"
        case "HydroSHED_15arc":
            dem_path = "WWF/HydroSHEDS/15CONDEM"
        case "GTOPO_30arc":
            dem_path = "USGS/GTOPO30"
        case _:
            Exception("INVALID DEM, see the options in the docstring")

    # Add the DEM as a Band.
    dem = ee.Image(dem_path).rename("ELEVATION")
    image = image.addBands(dem, None, True)

    # If the slope is required add it.
    if slope:
        slope = ee.Terrain.slope(dem).rename("SLOPE")
        image = image.addBands(slope, None, True)

    return image


def add_distance(
    image: ee.Image, features: ee.FeatureCollection, name: str = "DISTANCE"
) -> ee.Image:
    """
    Functio to add a band with the distance in meters from the features in a
    feature collections.

    Parameters
    ----------
    image : ee.Image
        Image of interest.

    features : ee.FeatureCollection
        Feature of interest.

    Returns
    -------
    image : ee.Image
        Image with a band of the distance raster from the features.
    """

    # Generate the distance raster.
    dist = features.distance().rename(name)

    # Add the distance raster as a band.
    image = image.addBands(dist, None, True)

    return image


def add_land_cover(image: ee.Image, year: int=2019) -> ee.Image:
    """
    Add Land Cover classification from COPERNICUS in a specific year.

    Parameters
    ----------
    image : ee.Image
        Image of interest.

    year : int, optional (2019)
        Year of the land cover image, select one from 2015 to 2019.

    Returns
    -------
    image : ee.Image
        Original image with land cover classifications as a band.
    """
    
    # Get land cover image
    land_cover = (
        ee.Image(f"COPERNICUS/Landcover/100m/Proba-V-C3/Global/{year}")
        .select("discrete_classification")
    )
    
    # Define old labels
    old_labels = [
        0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 111, 112, 113, 114, 115, 116, 
        121, 122, 123, 124, 126, 200
    ]

    # Define new labels
    new_labels = [i for i in range(len(old_labels))]

    # Change the old labels with the new labels and rename the band
    land_cover = land_cover.remap(old_labels, new_labels).rename("LAND_COVER")

    # Add land cover to the image
    image = image.addBands(land_cover, None, True)

    return image