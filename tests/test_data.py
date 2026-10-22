"""The prompt lists under data/ make the repo self-contained: no dataset
download (and no ``datasets`` dependency) at run time."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # abliterate.py imports torch at module level

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import abliterate  # noqa: E402


def test_data_covers_the_default_prompt_counts():
    defaults = abliterate.parse_args([])
    harmful_train, harmless_train, harmful_test = abliterate.load_prompts(
        defaults.n_train, defaults.n_test
    )
    assert len(harmful_train) == defaults.n_train
    assert len(harmless_train) == defaults.n_train
    assert len(harmful_test) == defaults.n_test
    every = harmful_train + harmless_train + harmful_test
    assert all(isinstance(t, str) and t.strip() for t in every)


def test_held_out_test_prompts_are_disjoint_from_train():
    harmful_train, _, harmful_test = abliterate.load_prompts(10_000, 10_000)
    assert not set(harmful_train) & set(harmful_test)
