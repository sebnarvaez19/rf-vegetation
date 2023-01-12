# Vegetation Coverage Classification on La Guajira (Colombia)

Classify the vegetation coverage of Cocientas basin in La Guajira (Colombia) with a Random Forest model to get the total vegetation coverage and try to predict futer changes.

## Folder Structure
```bash
.
├── data                        # Folder with all data used
│   ├── external                # Data recieved from external, not uploaded
│   ├── processed               # All processed data
│   ├── raster                  # Folder with the raster images, need to be created
│   │   └── train_test_image    # Image to train the model
│   └── shapefile               # Features used               
├── images                      # All final plots
├── maps                        # All final maps made in QGIS
├── src                         # All scripts made
│   └── functions               # Own functions to speed up processes
└── references                  
```

## Usage

You have to install the packages in [requirements.txt](https://github.com/srnarvaez/rf-vegetation/blob/4c51db7745b2c4bfcd44e4a6b0a2a8358e2e18a0/requirements.txt) file and their dependencies.

The original file with the control points will not be uploaded, but the [points sampled file](https://github.com/srnarvaez/rf-vegetation/blob/6b0621356d40287d0fed40cdbafe50a5be320637/data/processed/sample_points.geojson) is on processed folder, so the model could be runned.
