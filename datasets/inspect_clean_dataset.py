from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    path = Path(
        r"c:\Users\MarwanAl56ib\projects\phishguard\datasets\output\clean_urls_dataset.csv"
    )
    print("path:", path)
    exists = path.exists()
    print("exists:", exists)
    if not exists:
        return

    df = pd.read_csv(path)
    print("rows:", len(df))
    print("columns:", list(df.columns))

    print("\nlabel distribution:")
    print(df["label"].value_counts(dropna=False))

    print("\nmissing values per column:")
    print(df.isna().sum())

    print("\nfirst 5 rows:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

