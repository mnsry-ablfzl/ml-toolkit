from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
from scipy import stats
from scipy.stats import zscore
import umap
from statsmodels.graphics.mosaicplot import mosaic




def set_style(style: str = "weighted", figsize: tuple[int, int] = (10, 6)):
    """Set the default plotting style"""
    sns.set_theme(style=style)
    plt.rcParams["figure.figsize"] = figsize


def plot_missing_values(df: pd.Dataframe):
    """Plot missing values for each column"""
    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    if missing.empty:
        print("Mo missing Values found.")
        return
    
    plt.figure()
    sns.barplot(
        x=missing.values,
        y=missing.index
    )
    plt.title("Missing Values")
    plt.xlabel("Count")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(
        df: pd.DataFrame,
        annot: bool = True
):
    """Plot the correlation matrix"""
    numeric = df.select_dtypes(include="number")

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        numeric.corr(),
        annot=annot,
        cmap="coolwarm",
        square=True
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_histogram(
        df: pd.DataFrame,
        column: str,
        bins: int = 30
):
    """Plot a histogram"""

    plt.figure()
    sns.histplot(
        data=df,
        x=column,
        bins=bins,
        kde=True
    )

    plt.title(column)
    plt.tight_layout()
    plt.show()

def plot_boxplot(
        df: pd.DataFrame,
        column: str,
):
    """Plot a boxplot"""

    plt.figure()
    sns.boxplot(
        x=df[column],
    )

    plt.title(column)
    plt.tight_layout()
    plt.show()


def plot_countplot(
    df: pd.DataFrame,
    column: str,
):
    """Plot counts of categorical values."""

    plt.figure()

    sns.countplot(
        data=df,
        x=column,
    )

    plt.xticks(rotation=45)

    plt.title(column)
    plt.tight_layout()
    plt.show()


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
):
    """Scatter plot."""

    plt.figure()

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
    )

    plt.tight_layout()
    plt.show()


def plot_pairplot(
    df: pd.DataFrame,
    hue: str | None = None,
):
    """Pairplot of all numeric features."""

    sns.pairplot(
        df,
        hue=hue,
        diag_kind="kde",
    )

    plt.show()


def plot_kde(
    df: pd.DataFrame,
    column: str,
):
    """Kernel Density Estimation plot."""

    plt.figure()

    sns.kdeplot(
        df[column],
        fill=True,
    )

    plt.title(column)

    plt.tight_layout()
    plt.show()


def plot_violin(
    df: pd.DataFrame,
    x: str,
    y: str,
):
    """Violin plot."""

    plt.figure()

    sns.violinplot(
        data=df,
        x=x,
        y=y,
    )

    plt.tight_layout()
    plt.show()

def plot_feature_distributions(
    df: pd.DataFrame,
    bins: int = 30,
    figsize: tuple[int, int] = (15, 10),
):
    """
    Plot the distribution of every numerical feature.
    """

    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        raise ValueError("No numerical columns found.")

    n_cols = 3
    n_rows = math.ceil(len(numeric.columns) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
    )

    axes = axes.flatten()

    for ax, column in zip(axes, numeric.columns):
        sns.histplot(
            data=numeric,
            x=column,
            kde=True,
            bins=bins,
            ax=ax,
        )

        ax.set_title(column)

    for ax in axes[len(numeric.columns):]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


def plot_ecdf(
    df: pd.DataFrame,
    column: str,
):
    """
    Plot the empirical cumulative distribution function.
    """

    plt.figure(figsize=(8, 5))

    sns.ecdfplot(
        data=df,
        x=column,
    )

    plt.title(f"ECDF of {column}")
    plt.xlabel(column)
    plt.ylabel("Cumulative Probability")

    plt.tight_layout()
    plt.show()


def plot_qq(
    df: pd.DataFrame,
    column: str,
):
    """
    Q-Q plot for testing normality.
    """

    plt.figure(figsize=(6, 6))

    stats.probplot(
        df[column].dropna(),
        dist="norm",
        plot=plt,
    )

    plt.title(f"Q-Q Plot ({column})")

    plt.tight_layout()
    plt.show()

def plot_jointplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    kind: str = "scatter",
    hue: str | None = None,
    height: float = 8,
):
    """
    Plot a joint distribution of two variables.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
        X-axis column.
    y : str
        Y-axis column.
    kind : {"scatter", "kde", "hist", "hex", "reg"}
    hue : str | None
        Optional categorical variable for coloring (only supported for scatter).
    height : float
        Figure size.
    """

    sns.jointplot(
        data=df,
        x=x,
        y=y,
        kind=kind,
        hue=hue,
        height=height,
    )

    plt.show()


def plot_hexbin(
    df: pd.DataFrame,
    x: str,
    y: str,
    gridsize: int = 30,
    cmap: str = "viridis",
):
    """
    Plot a hexbin chart.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
    y : str
    gridsize : int
    cmap : str
    """

    plt.figure(figsize=(8, 6))

    plt.hexbin(
        df[x],
        df[y],
        gridsize=gridsize,
        cmap=cmap,
    )

    plt.colorbar(label="Count")

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y}")

    plt.tight_layout()
    plt.show()

def plot_regression(
    df: pd.DataFrame,
    x: str,
    y: str,
    ci: int | None = 95,
):
    """
    Plot a regression line.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
    y : str
    ci : int | None
        Confidence interval.
    """

    plt.figure(figsize=(8, 6))

    sns.regplot(
        data=df,
        x=x,
        y=y,
        ci=ci,
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "red"},
    )

    plt.title(f"{y} vs {x}")

    plt.tight_layout()
    plt.show()


def plot_stacked_bar(
    df: pd.DataFrame,
    x: str,
    hue: str,
    normalize: bool = False,
    figsize: tuple[int, int] = (8, 6),
):
    """
    Plot a stacked bar chart for two categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
        Main categorical variable.
    hue : str
        Variable used for stacking.
    normalize : bool, default=False
        If True, plot proportions instead of counts.
    figsize : tuple[int, int], default=(8, 6)
    """

    table = pd.crosstab(
        df[x],
        df[hue],
        normalize="index" if normalize else False,
    )

    ax = table.plot(
        kind="bar",
        stacked=True,
        figsize=figsize,
    )

    ax.set_xlabel(x)
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title(f"{x} by {hue}")

    plt.xticks(rotation=45)
    plt.legend(title=hue)

    plt.tight_layout()
    plt.show()


def plot_mosaic(
    df: pd.DataFrame,
    x: str,
    y: str,
    figsize: tuple[int, int] = (10, 6),
):
    """
    Plot a mosaic plot for two categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
    y : str
    figsize : tuple[int, int], default=(10, 6)
    """

    plt.figure(figsize=figsize)

    mosaic(
        df,
        [x, y],
        gap=0.02,
    )

    plt.title(f"{x} vs {y}")

    plt.tight_layout()
    plt.show()

def plot_outliers(
    df: pd.DataFrame,
    column: str,
):
    """
    Plot a boxplot to visualize outliers.
    """

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x=df[column],
    )

    plt.title(f"Outliers in {column}")

    plt.tight_layout()
    plt.show()

def plot_zscore(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.0,
):
    """
    Visualize outliers using the Z-score method.
    """

    values = df[column].dropna()

    z = np.abs(zscore(values))

    plt.figure(figsize=(10, 5))

    plt.scatter(
        values.index,
        values,
        alpha=0.7,
        label="Data",
    )

    plt.scatter(
        values.index[z > threshold],
        values[z > threshold],
        color="red",
        label="Outliers",
    )

    plt.axhline(values.mean(), color="green", linestyle="--")

    plt.title(f"Z-score Outliers ({column})")
    plt.xlabel("Index")
    plt.ylabel(column)

    plt.legend()

    plt.tight_layout()
    plt.show()

    return df.loc[values.index[z > threshold]]


def plot_iqr(
    df: pd.DataFrame,
    column: str,
):
    """
    Visualize outliers using the IQR method.
    """

    values = df[column]

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (values < lower) | (values > upper)

    plt.figure(figsize=(10, 5))

    plt.scatter(
        values.index,
        values,
        alpha=0.7,
        label="Data",
    )

    plt.scatter(
        values.index[mask],
        values[mask],
        color="red",
        label="Outliers",
    )

    plt.axhline(lower, color="orange", linestyle="--")
    plt.axhline(upper, color="orange", linestyle="--")

    plt.title(f"IQR Outliers ({column})")
    plt.xlabel("Index")
    plt.ylabel(column)

    plt.legend()

    plt.tight_layout()
    plt.show()

    return df.loc[mask]


def plot_timeseries(
    df: pd.DataFrame,
    x: str,
    y: str,
    figsize: tuple[int, int] = (12, 6),
):
    """
    Plot a time series.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
        Datetime column.
    y : str
        Numeric column.
    figsize : tuple[int, int], default=(12, 6)
    """

    data = df.copy()
    data[x] = pd.to_datetime(data[x])

    plt.figure(figsize=figsize)

    plt.plot(
        data[x],
        data[y],
        linewidth=2,
    )

    plt.title(f"{y} Over Time")
    plt.xlabel(x)
    plt.ylabel(y)

    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_rolling_mean(
    df: pd.DataFrame,
    x: str,
    y: str,
    window: int = 7,
    figsize: tuple[int, int] = (12, 6),
):
    """
    Plot a rolling mean.

    Parameters
    ----------
    window : int
        Rolling window size.
    """

    data = df.copy()
    data[x] = pd.to_datetime(data[x])

    data = data.sort_values(x)

    rolling = data[y].rolling(window).mean()

    plt.figure(figsize=figsize)

    plt.plot(
        data[x],
        data[y],
        alpha=0.4,
        label="Original",
    )

    plt.plot(
        data[x],
        rolling,
        linewidth=2,
        label=f"{window}-Period Rolling Mean",
    )

    plt.legend()

    plt.title(f"Rolling Mean ({window})")

    plt.tight_layout()
    plt.show()

def plot_seasonality(
    df: pd.DataFrame,
    x: str,
    y: str,
    period: str = "month",
    figsize: tuple[int, int] = (10, 5),
):
    """
    Plot seasonal averages.

    Parameters
    ----------
    period : {"month", "dayofweek", "day", "quarter", "year"}
    """

    data = df.copy()
    data[x] = pd.to_datetime(data[x])

    if period == "month":
        groups = data[x].dt.month
    elif period == "dayofweek":
        groups = data[x].dt.day_name()
    elif period == "day":
        groups = data[x].dt.day
    elif period == "quarter":
        groups = data[x].dt.quarter
    elif period == "year":
        groups = data[x].dt.year
    else:
        raise ValueError(f"Unsupported period: {period}")

    seasonal = data.groupby(groups)[y].mean()

    plt.figure(figsize=figsize)

    seasonal.plot(
        kind="bar",
    )

    plt.title(f"{y} by {period.capitalize()}")
    plt.xlabel(period.capitalize())
    plt.ylabel(f"Average {y}")

    plt.tight_layout()
    plt.show()

def plot_pca(
    df: pd.DataFrame,
    target: str | None = None,
    n_components: int = 2,
    figsize: tuple[int, int] = (8, 6),
):
    """
    Plot PCA projection.

    Parameters
    ----------
    df : pd.DataFrame
    target : str | None
        Target column used for coloring.
    n_components : int
        Number of principal components.
    """

    data = df.copy()

    if target is not None:
        y = data[target]
        X = data.drop(columns=target)
    else:
        y = None
        X = data

    X = X.select_dtypes(include="number")

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components)

    components = pca.fit_transform(X_scaled)

    plt.figure(figsize=figsize)

    if y is not None:
        scatter = plt.scatter(
            components[:, 0],
            components[:, 1],
            c=pd.Categorical(y).codes,
            cmap="tab10",
            alpha=0.8,
        )

        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=y.unique(),
            title=target,
        )

    else:
        plt.scatter(
            components[:, 0],
            components[:, 1],
            alpha=0.8,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(
        f"PCA ({pca.explained_variance_ratio_.sum():.2%} Variance Explained)"
    )

    plt.tight_layout()
    plt.show()


def plot_tsne(
    df: pd.DataFrame,
    target: str | None = None,
    perplexity: int = 30,
    random_state: int = 42,
    figsize: tuple[int, int] = (8, 6),
):
    """
    Plot t-SNE embedding.
    """

    data = df.copy()

    if target is not None:
        y = data[target]
        X = data.drop(columns=target)
    else:
        y = None
        X = data

    X = X.select_dtypes(include="number")

    X = StandardScaler().fit_transform(X)

    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
    ).fit_transform(X)

    plt.figure(figsize=figsize)

    if y is not None:
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=pd.Categorical(y).codes,
            cmap="tab10",
            alpha=0.8,
        )

        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=y.unique(),
            title=target,
        )

    else:
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
        )

    plt.title("t-SNE Projection")

    plt.tight_layout()
    plt.show()


def plot_umap(
    df: pd.DataFrame,
    target: str | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    figsize: tuple[int, int] = (8, 6),
):
    """
    Plot UMAP embedding.
    """

    data = df.copy()

    if target is not None:
        y = data[target]
        X = data.drop(columns=target)
    else:
        y = None
        X = data

    X = X.select_dtypes(include="number")

    X = StandardScaler().fit_transform(X)

    embedding = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    ).fit_transform(X)

    plt.figure(figsize=figsize)

    if y is not None:
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=pd.Categorical(y).codes,
            cmap="tab10",
            alpha=0.8,
        )

        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=y.unique(),
            title=target,
        )

    else:
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
        )

    plt.title("UMAP Projection")

    plt.tight_layout()
    plt.show()




