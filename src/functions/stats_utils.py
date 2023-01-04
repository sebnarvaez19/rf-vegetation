import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA


def plot_variable_importance(
    X: ArrayLike, 
    y: pd.Series, 
    labels: pd.Series | None = None, 
    ax: Axes | None = None,
) -> Axes:
    """
    Plot a PCA with the two forst component to show what variables in X are more
    important to define y.

    Paremeters
    ----------
    X : ArrayLike
        Independent variables.

    y : pandas.Series
        Dependent variable.

    labels : pandas.Series | None (Optional) = None
        Labels for dependent variable.

    ax : matplotlib.axes.Axes | None (optional) = None
        Axes to draw the plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Scatter of PCA.
    """

    if labels is None:
        labels = y

    # Calculate the PCA and extract the components and values
    pca = PCA(n_components=2).fit(X)
    com = pca.components_.T
    val = pca.transform(X)

    # Guides for plot
    xc = np.linspace(-0.5, 0.5, 500)
    xcp = np.sqrt(0.5**2 - xc**2)
    xcn = -np.sqrt(0.5**2 - xc**2)

    if ax == None:
        # Create figure
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Add guides
    ax.axhline(0, color="black", lw=0.2)
    ax.axvline(0, color="black", lw=0.2)
    ax.plot(xc, xcp, color="red", lw=0.2)
    ax.plot(xc, xcn, color="red", lw=0.2)

    # Add values by y
    for yi, label in zip(y.unique(), labels.unique()):
        mask = y == yi
        ax.scatter(val[mask, 0], val[mask, 1], alpha=0.4, label=label)

    # Add vectors to show the importnace of the variables in X
    for var, delta in zip(X.columns, com):
        ax.arrow(0, 0, delta[0], delta[1], color="black")
        ax.text(delta[0], delta[1], var, va="center", ha="center", size=6)

    # Set lims and labels
    ax.set(xlim=(-0.7, 0.7), ylim=(-0.7, 0.7))
    ax.set_xlabel(f"Component 1: {pca.explained_variance_ratio_[1]:0.3f}")
    ax.set_ylabel(f"Component 2: {pca.explained_variance_ratio_[0]:0.3f}")

    # Add legend in uper-right corner
    ax.legend(loc=1)

    return ax


def group_boxplot(
    data: pd.DataFrame,
    variables: ArrayLike,
    y: str,
    title: str | None = None,
    ticklabels: ArrayLike | None = None,
    showfliers: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """
    Function to melt variable and plot it on a boxplot.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with the variables of interest.

    variables : ArrayLike
        Variables to merge.

    y : str
        Variable to group.

    title : str | None (Optional) = None
        Title for the plot.

    ticklabels : ArrayLike | None (optional) = None
        Ticklabels for plot.

    showfliers : bool (optional) = True
        Show outliers in boxplot.

    ax : matplotlib.axes.Axes | None (optional) = None
        Axes to draw the plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Boxplot.
    """

    # Melt data to long format
    df = pd.melt(data, id_vars=y, value_vars=variables)

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # Plot data
    sns.boxplot(data=df, y="value", x="variable", hue=y, showfliers=showfliers, ax=ax)

    # Hide unnecesary label
    ax.set_xlabel(None)

    # Hide legend by default
    ax.legend().set_visible(False)

    # If there is a title, use it
    if title != None:
        ax.set_title(title)

    # If there are ticklabels, use them
    if ticklabels != None:
        ax.set_xticklabels(ticklabels)

    return ax
