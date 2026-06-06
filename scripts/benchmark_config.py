"""Benchmark patient manifest (shared by the py sweep, aggregate, figure).

The 17 3CA patients are the same frozen slices used by the sibling pycopykat /
pyinfercnv benchmarks. Per the workspace convention we *reference* those slices
rather than copy them: ``--data-root`` defaults to ``../pycopykat/benchmarks/
full`` relative to this repo, so no absolute paths are baked into committed code.

Each patient dir holds ``counts.tsv`` (genes x cells) + ``cells.csv`` (3CA
annotation, layout only -- not a truth label).
"""
from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT.parent / "pycopykat" / "benchmarks" / "full"

# (cancer_label, [patient, ...]) -- mirrors pycopykat's run_all_benchmarks.
DATASETS: list[tuple[str, list[str]]] = [
    ("Gao2021_Breast", ["DCIS1", "TNBC1", "TNBC2", "TNBC3"]),
    ("Kim2020_Lung", ["P1028", "P0019", "P0034"]),
    ("Lee2020_Colorectal", ["SMC16", "SMC09", "SMC21"]),
    ("Obradovic2021_Kidney", ["Patient4", "Patient5", "Patient2"]),
    ("Qian2020_Ovarian", ["11", "14", "12", "13"]),
]


def iter_patients(data_root: Path):
    """Yield (label, patient, counts_path, cells_path) for every patient."""
    for label, patients in DATASETS:
        for pt in patients:
            d = data_root / label / pt
            yield label, pt, d / "counts.tsv", d / "cells.csv"


def all_sam_names() -> list[str]:
    return [f"{label}__{pt}" for label, patients in DATASETS for pt in patients]
