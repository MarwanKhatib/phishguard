from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests


OPENPHISH_FEED_URL = (
    "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
)


def normalize_url(raw: str) -> Optional[str]:
    """Normalize a raw URL or domain string.

    Args:
        raw: Raw value from input datasets.

    Returns:
        Normalized URL string or None if the input is empty.
    """
    value = (raw or "").strip()
    if not value:
        return None
    if value.startswith("`") and value.endswith("`"):
        value = value[1:-1].strip()
    if " " in value:
        parts = value.split()
        value = parts[-1].strip()
    if "://" not in value:
        value = f"https://{value}"
    return value


def load_benign_links(path: Path) -> pd.DataFrame:
    """Load and clean benign links from a CSV file.

    The file is expected to contain at least a 'url' column with domains or
    URLs. All rows are labeled as benign (0).

    Args:
        path: Path to the benign links CSV file.

    Returns:
        DataFrame with columns: url, label, source.
    """
    if not path.exists():
        return pd.DataFrame(columns=["url", "label", "source"])

    df = pd.read_csv(path, encoding="utf-8")
    if "url" not in df.columns:
        return pd.DataFrame(columns=["url", "label", "source"])

    urls: List[str] = []
    for value in df["url"].astype(str):
        url = normalize_url(value)
        if url is not None:
            urls.append(url)

    if not urls:
        return pd.DataFrame(columns=["url", "label", "source"])

    cleaned = pd.DataFrame(
        {
            "url": urls,
            "label": [0] * len(urls),
            "source": ["benign_links"] * len(urls),
        }
    )
    return cleaned


def load_phiusiil_dataset(path: Path) -> pd.DataFrame:
    """Load and clean the PhiUSIIL phishing URL dataset.

    This dataset contains a 'URL' column and a 'label' column where:
    - 1 denotes benign.
    - 0 denotes phishing.

    The function inverts the labels so that:
    - 1 denotes phishing.
    - 0 denotes benign.

    Args:
        path: Path to the PhiUSIIL CSV file.

    Returns:
        DataFrame with columns: url, label, source.
    """
    if not path.exists():
        return pd.DataFrame(columns=["url", "label", "source"])

    df = pd.read_csv(path, encoding="utf-8", engine="python")
    if "URL" not in df.columns or "label" not in df.columns:
        return pd.DataFrame(columns=["url", "label", "source"])

    urls: List[str] = []
    labels: List[int] = []

    for _, row in df.iterrows():
        raw_url = str(row["URL"])
        url = normalize_url(raw_url)
        if url is None:
            continue
        try:
            raw_label = int(row["label"])
        except Exception:
            continue
        label = 1 - raw_label
        if label not in (0, 1):
            continue
        urls.append(url)
        labels.append(label)

    if not urls:
        return pd.DataFrame(columns=["url", "label", "source"])

    cleaned = pd.DataFrame(
        {
            "url": urls,
            "label": labels,
            "source": ["phiusiil"] * len(urls),
        }
    )
    return cleaned


def load_openphish_feed(feed_url: str = OPENPHISH_FEED_URL) -> pd.DataFrame:
    """Load and clean URLs from the OpenPhish public feed.

    Args:
        feed_url: URL of the OpenPhish feed.

    Returns:
        DataFrame with columns: url, label, source.
    """
    try:
        response = requests.get(feed_url, timeout=10.0)
        response.raise_for_status()
    except Exception:
        return pd.DataFrame(columns=["url", "label", "source"])

    urls: List[str] = []
    for line in response.text.splitlines():
        url = normalize_url(line)
        if url is None:
            continue
        urls.append(url)

    if not urls:
        return pd.DataFrame(columns=["url", "label", "source"])

    cleaned = pd.DataFrame(
        {
            "url": urls,
            "label": [1] * len(urls),
            "source": ["openphish"] * len(urls),
        }
    )
    return cleaned


def build_clean_dataset(input_dir: Path, output_path: Path) -> None:
    """Build a cleaned, merged URL dataset from input CSV files.

    The following inputs are supported:
    - begnin_links.csv: benign domains/URLs with header 'url'.
    - PhiUSIIL_Phishing_URL_Dataset.csv: PhiUSIIL dataset with 'URL' and 'label'.
    - OpenPhish public feed (downloaded at runtime).

    The resulting CSV contains:
    - url: normalized URL string.
    - label: 0 for benign, 1 for phishing.
    - source: origin of the entry (benign_links, phiusiil, openphish).

    Args:
        input_dir: Directory containing the input CSV files.
        output_path: Path of the resulting cleaned CSV file.
    """
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)

    benign_path = input_dir / "begnin_links.csv"
    phiusiil_path = input_dir / "PhiUSIIL_Phishing_URL_Dataset.csv"

    benign_df = load_benign_links(benign_path)
    phiusiil_df = load_phiusiil_dataset(phiusiil_path)
    openphish_df = load_openphish_feed()

    frames = [benign_df, phiusiil_df, openphish_df]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["url"])
    merged = merged.drop_duplicates(subset=["url"])
    merged = merged[["url", "label"]]

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)


def main() -> None:
    """Entry point for building the cleaned URL dataset."""
    root = Path(__file__).resolve().parent.parent
    input_dir = root / "datasets" / "input"
    output_path = root / "datasets" / "output" / "clean_urls_dataset.csv"
    build_clean_dataset(input_dir=input_dir, output_path=output_path)


if __name__ == "__main__":
    main()
