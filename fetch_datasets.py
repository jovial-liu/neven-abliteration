#!/usr/bin/env python3
"""Regenerate the prompt lists in data/ from the Hugging Face Hub.

The CSVs in data/ are committed with the repo, so abliterate.py needs no dataset
download at run time. This script only exists to record where they came from and
to re-fetch them if they ever change:

    pip install datasets
    python fetch_datasets.py

Sources (the same prompt sets used in Maxime Labonne's abliteration post):
    mlabonne/harmful_behaviors    train + test splits, column "text"
    mlabonne/harmless_alpaca      train split, column "text"
"""

from __future__ import annotations

import csv
from pathlib import Path

from datasets import load_dataset

DATA = Path(__file__).resolve().parent / "data"

SOURCES = {
    "harmful_behaviors_train.csv": ("mlabonne/harmful_behaviors", "train"),
    "harmful_behaviors_test.csv": ("mlabonne/harmful_behaviors", "test"),
    "harmless_alpaca.csv": ("mlabonne/harmless_alpaca", "train"),
}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for filename, (repo, split) in SOURCES.items():
        texts = load_dataset(repo, split=split)["text"]
        path = DATA / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            writer.writerows([[t] for t in texts])
        print(f"{path.name}: {len(texts)} prompts")


if __name__ == "__main__":
    main()
