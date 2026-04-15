from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_dataframe(path: Path, input_format: str = "jsonl") -> pd.DataFrame:
    fmt = input_format.lower()
    if fmt == "jsonl":
        return pd.DataFrame(load_jsonl(path))
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported input format: {input_format}")


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
