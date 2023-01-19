import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import NullLocator
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA


def corr_matrix(
    data: pd.DataFrame,
    variables: ArrayLike | None = None,
    half: bool = False,
    hide_insignificants: bool = False,
    singificant_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate the pearson correlation matrix of the variables in a dataframe.
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with the variables to evaluate their correlation.
    variables : ArrayLike | None = None
        The variables of interest, if it is not defined, all variables in
        the dataframe will be evaluated.
    half : bool = False
        If True, only show the corerlation of the first half of the matrix,
        excluding the repeated correlation.
    hide_insignifcants : bool = False
        If True, hide all the correlation with a p-value greater than the
        significant threshold.
    siginificant_threshold : float = 0.05
        Threshold of significant correlation.
    returns
    -------
    corr : pd.DataFrame
        Dataframe with the correlation values.
    """
    if variables == None:
        variables = data.columns

    reverse = variables[::-1]

    N = len(variables)

    corr = np.empty((N, N))
    pval = np.full((N, N), np.nan)
    mask = np.full((N, N), np.nan)

    for i, iv in enumerate(variables):
        for j, jv in enumerate(reverse):
            c, p = pearsonr(data[iv], data[jv])
            corr[j, i] = c

            if p <= singificant_threshold:
                pval[j, i] = 1.0

        mask[: N - i, i] = 1.0

    if half:
        corr *= mask

    if hide_insignificants:
        corr *= pval

    corr = pd.DataFrame(data=corr, index=reverse, columns=variables)

    return corr


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
    ax.legend(loc=1, title=labels.name)

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


def plot_corr_matrix(
    data: pd.DataFrame,
    variables: ArrayLike | None = None,
    half: bool = False,
    hide_insignificants: bool = False,
    significant_threshold: float = 0.05,
    show_labels: bool = True,
    show_colorbar: bool = False,
    palette: str = "Spectral",
    text_color: str = "black",
    ax: Axes | None = None,
) -> Axes:
    """
    Calculate the pearson correlation matrix of the variables in a dataframe.
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with the variables to evaluate their correlation.
    variables : ArrayLike | None = None
        The variables of interest, if it is not defined, all variables in
        the dataframe will be evaluated.
    half : bool = False
        If True, only show the corerlation of the first half of the matrix,
        excluding the repeated correlation.
    hide_insignifcants : bool = False
        If True, hide all the correlation with a p-value greater than the
        significant threshold.
    significant_threshold : float = 0.05
        Threshold of significant correlation.
    show_labels : bool = True
        Show the correlation value.
    show_colorbar : bool = False
        Show colorbar.
    palette : str = Spectral
        Color palette for correlation plot.
    text_color : str = black
        Color of text correlation labels.
    ax : matplotlib.axes.Axes | None = None
        Axes to draw the correlation matrix.
    returns
    -------
    ax : matplotlib.axes.Axes
        Correltion matrix.
    """
    # If variables are not defined get all columns from data
    if variables == None:
        variables = data.columns

    # Get the number of variables
    N = len(variables)

    # Reverse variables for plot
    reverse = variables[::-1]

    # Get the correlation matrix
    corr = corr_matrix(
        data, variables, half, hide_insignificants, significant_threshold
    )

    # If there not axes create one
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Plot matrix with pcolormesh
    im = ax.pcolormesh(
        variables, reverse, corr, cmap=palette, edgecolor="w", vmin=-1, vmax=1
    )

    # Invert y axis
    ax.invert_yaxis()

    # Add the colorbar
    if show_colorbar:
        cax = ax.inset_axes([1.04, 0.1, 0.05, 0.8])
        bar = plt.colorbar(im, cax=cax, label="Correlation")

    if show_labels:
        x, y = np.meshgrid(np.arange(N), np.arange(N))
        x = x.reshape(-1)
        y = y.reshape(-1)
        t = corr.values.reshape(-1)

        for xi, yi, ti in zip(x, y, t):
            if np.isfinite(ti):
                ax.text(
                    xi, yi, 
                    round(ti, 2), 
                    color=text_color, 
                    size=8, 
                    ha="center", 
                    va="center"
                )

    # Rotate labels to improve their readability
    ax.set_xticklabels(variables, rotation=30, ha="right")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

    return ax
