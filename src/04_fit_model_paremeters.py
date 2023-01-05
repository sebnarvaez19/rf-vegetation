import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.style.use("src/style.mplstyle")

seed = 1998
size = 0.4
variables = [
    "GREEN", "NIR", "SWIR", "TEMPERATURE", 
    "NDVI", "ELEVATION", "DISTANCE_SHORELINE",
]

data_path = "data/processed/sample_points.geojson"
save_image_path = "images/{:02d}_{}.svg"

def main():
    # Read data
    gdf = gpd.read_file(data_path)

    X = gdf[variables].copy()
    y = gdf["vegetation"].copy()

    X = StandardScaler().fit_transform(X)

    # Split in train an test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=size, random_state=seed
    )

    # Check how many trees are needed 
    accuracies = []
    for i in range(2, 100+1):
        rf = RandomForestClassifier(n_estimators=i, n_jobs=-1)
        rf.fit(X_train, y_train)

        y_train_pred = rf.predict(X_train)
        accuracies.append(accuracy_score(y_train, y_train_pred))

    accuracies = np.array(accuracies)

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(np.arange(2, accuracies.shape[0]+2), accuracies)
    ax.set(xlabel="Tree number", ylabel="Accuracy")
    fig1.savefig(save_image_path.format(4, "n_trees_efect"))

    # Check how the maximum depth of the trees impact the model
    accuracies = []
    for i in range(2, 100+1):
        rf = RandomForestClassifier(n_estimators=30, max_leaf_nodes=i, n_jobs=-1)
        rf.fit(X_train, y_train)

        y_train_pred = rf.predict(X_train)
        accuracies.append(accuracy_score(y_train, y_train_pred))

    accuracies = np.array(accuracies)

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.plot(np.arange(2, accuracies.shape[0]+2), accuracies)
    ax.set(xlabel="Max nodes number", ylabel="Accuracy")
    fig2.savefig(save_image_path.format(5, "max_n_node_effect"))

    # Show variable importance
    rf = RandomForestClassifier(
        n_estimators=30, max_leaf_nodes=20, n_jobs=-1, oob_score=True
    )

    rf.fit(X_train, y_train)

    importance = pd.DataFrame({
        "variable": [v.capitalize()[:5] for v in variables],
        "importance": rf.feature_importances_
    }).sort_values(by="importance", ascending=False)

    fig3 = plt.figure()
    ax = fig3.add_subplot(111)
    ax.bar(importance.variable, importance.importance.cumsum())
    ax.set(xlabel="Variables", ylabel="Cumulative importance")
    fig3.savefig(save_image_path.format(6, "cumulative_variable_importance"))

    # Predict
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # Print scores
    print(f"OOB Score: {rf.oob_score_:0.3f}")
    print(f"Train accuracy: {accuracy_score(y_train, y_train_pred):0.3f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_test_pred):0.3f}")

    print("Train F1 score: ", f1_score(y_train, y_train_pred, average="macro"), sep="\n")
    print("Test F1 score: ", f1_score(y_test, y_test_pred, average="macro"), sep="\n")

    # Print confussion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    print("Confussion matrix for Train: ", cm_train, sep="\n")
    print("Confussion matrix for Test: ", cm_test, sep="\n")

    plt.show()

    return None

if __name__ == "__main__":
    main()