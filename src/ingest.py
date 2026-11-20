"""Stage 1: fetch the raw dataset and assert its shape.

Run: python -m src.ingest
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import DATA_URL, RAW, RAW_SHA256

EXPECTED_MIN_ROWS = 700
REQUIRED_COLS = {"dteday", "cnt", "casual", "registered", "temp", "hum", "windspeed"}


def verify_checksum(path=RAW) -> str:
    """Fail unless the raw file is byte-identical to the one we validated."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != RAW_SHA256:
        expected_prefix = RAW_SHA256[:16]
        actual_prefix = digest[:16]
        raise ValueError(
            f"sha256 mismatch: expected {expected_prefix}, got {actual_prefix}. "
            "Restore it with `git checkout -- data/raw/`."
        )
    return digest


def download(force: bool = False) -> None:
    if RAW.exists() and not force:
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, RAW)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, parse_dates=["dteday"])
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"raw data is missing required columns: {sorted(missing)}")
    if len(df) < EXPECTED_MIN_ROWS:
        raise ValueError(f"raw data has only {len(df)} rows; expected >= {EXPECTED_MIN_ROWS}")
    return df


if __name__ == "__main__":
    download(force="--force" in sys.argv)
    digest = verify_checksum(RAW)
    df = load_raw()
    print(f"OK ingest: {len(df)} rows, {df.dteday.min().date()} -> {df.dteday.max().date()}, sha256={digest[:16]}")
