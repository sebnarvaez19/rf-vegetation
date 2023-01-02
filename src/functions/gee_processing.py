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
    # Define the original and the new names
    ori_names = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "ST_B6"]
    new_names = ["BLUE", "GREEN", "RED", "NIR", "TEMPERATURE"]

    # Select the bands with the original names an rename them
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

    indices = {
        "NDVI": "(b('NIR') - b('RED'))/(b('NIR') + b('RED'))",
        "EVI": "2.5*((b('NIR') - b('RED'))/ \
                (b('NIR') + 6*b('RED') - 7.5*b('BLUE') + 1))",
        "NDMI": "(b('NIR') - b('RED'))/(b('NIR') + b('RED'))",
        "SAVI": "((b('NIR') - b('RED'))/(b('NIR') + b('RED') + 0.5))*(1.5)",
        "GCI": "(b('NIR')/b('GREEN')) - 1",
    }

    for name, formula in indices.items():
        image = image.addBands(image.expression(formula).toFloat().rename(name))

    return image
