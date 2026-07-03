from collections.abc import Sequence

import pandas as pd


def value_counts(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    dropna: bool = False,
) -> pd.DataFrame:
    """
    Return value counts for one or more columns as a tidy DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : Sequence[str] | None, default=None
        Columns to summarize. If None, all columns are used.
    dropna : bool, default=False
        Whether to exclude missing values.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - column
        - value
        - count
        - percent
    """
    if columns is None:
        columns = list(df.columns)

    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Columns not found: {', '.join(sorted(missing))}")

    result = []

    for column in columns:
        counts = df[column].value_counts(dropna=dropna)
        percentages = df[column].value_counts(
            normalize=True,
            dropna=dropna,
        ).mul(100)

        summary = pd.DataFrame(
            {
                "column": column,
                "value": counts.index,
                "count": counts.values,
                "percent": percentages.values.round(2),
            }
        )

        result.append(summary)

    return pd.concat(result, ignore_index=True)