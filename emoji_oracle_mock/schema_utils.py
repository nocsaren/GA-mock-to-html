from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def read_csv_header(path: Path) -> list[str] | None:
    if not path.exists() or not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return None


def ensure_columns(df, columns: Iterable[str]):
    for c in columns:
        if c not in df.columns:
            df[c] = None
    return df[columns]
