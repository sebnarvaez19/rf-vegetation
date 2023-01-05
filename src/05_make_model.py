import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def main():
    seed = 1998

    class DropNAN(BaseEstimator, TransformerMixin):
    
        def fit(self, X, y=None):

            return self

        def transform(self, X):
            if isinstance(X, pd.DataFrame):
                X = X.values

            valids = np.isfinite(X).all(axis=1)

            return X[valids,:]

    preprocessing = Pipeline([
        ("Drop NaN rows", DropNAN()),
        ("Sacler", StandardScaler()),
    ])

    model = Pipeline([
        ("Preprocessing", preprocessing),
        ("Classifier", RandomForestClassifier(
            n_estimators=30, 
            max_leaf_nodes=20, 
            n_jobs=-1, 
            oob_score=True, 
            random_state=seed
        ))
    ])

    print(model)

    return None

if __name__ == "__main__":
    main()
