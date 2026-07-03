from pathlib import Path

import pandas as pd


def load_csv(
    path: str | Path,
    *,
    preview_rows: int = 5,
    show_summary: bool = True,
) -> pd.DataFrame:
    """
    Load a CSV file and optionally display a dataset summary.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    preview_rows : int, default=5
        Number of rows to display.
    show_summary : bool, default=True
        Whether to print a summary.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    df = pd.read_csv(path)

    if show_summary:
        print("\n" + "=" * 60)
        print("📊 DATASET OVERVIEW")
        print("=" * 60)

        print(f"Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Memory     : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print("\n📋 Columns")
        print("-" * 60)
        for col in df.columns:
            print(f"• {col}")

        print("\n🔍 Data Types")
        print("-" * 60)
        print(df.dtypes)

        print("\n👀 Preview")
        print("-" * 60)
        print(df.head(preview_rows))

        print("\n❓ Missing Values")
        print("-" * 60)
        missing = df.isna().sum()
        missing = missing[missing > 0]

        if len(missing):
            print(missing.sort_values(ascending=False))
        else:
            print("No missing values found.")

    return df