import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Set random seed
seed = 1998

# Load train data
data = gpd.read_file("data/processed/sample_points.geojson", index_col=0)
variables = [
    "GREEN",
    "NIR",
    "SWIR",
    "TEMPERATURE",
    "ELEVATION",
    "DISTANCE_SHORELINE",
    "NDVI",
]

X = data[variables].copy()
y = data["vegetation"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=seed
)

# Define a estimator to drop the NaNs
class DropNAN(BaseEstimator, TransformerMixin):
    """
    Estimator to drop NaN values in X data.
    """

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        """
        Remove NaN from a dataframe or array.

        Parameters
        ----------
        X : pd.DataFrame | numpy.Array
            X data.

        Returns
        -------
        X : numpy.Array
            X data without NaNs.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        valids = np.isfinite(X).all(axis=1)

        return X[valids, :]


# Piple for preprocessing data
preprocessing = Pipeline(
    [
        ("Drop NaN rows", DropNAN()),
        ("Sacler", StandardScaler()),
    ]
)

# Final model
final_model = Pipeline(
    [
        ("Preprocessing", preprocessing),
        (
            "Classifier",
            RandomForestClassifier(
                n_estimators=30,
                max_leaf_nodes=20,
                n_jobs=-1,
                oob_score=True,
                random_state=seed,
            ),
        ),
    ]
)

rf_classifier = final_model.fit(X_train, y_train)
