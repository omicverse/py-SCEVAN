"""Per-patient pyscevan vs R SCEVAN comparison -> metrics.csv.

Explicit file paths so the same script serves both the 3CA patients (R
outputs from ``run_r_scevan.R``) and the frozen MGH106 reference
(``tests/r_ref/mgh106``). Measures py<->R consistency only (no external
truth). See ``bench_metrics`` for the metric definitions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bench_metrics import cna_spearman, speedup, tumor_consistency

_ANNOT_COLS = ("gene_id", "seqnames", "end")


def _read_cna(path: Path) -> pd.DataFrame:
    """Read a CNAmat TSV, tolerating an R ``col.names=NA`` leading index col."""
    compression = "gzip" if str(path).endswith(".gz") else None
    df = pd.read_csv(path, sep="\t", compression=compression)
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=df.columns[0])
    return df


def _read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in open(path) if ln.strip()]


def _read_elapsed_sec(path: Path) -> float:
    if not path or not path.exists():
        return float("nan")
    for ln in path.read_text().splitlines():
        if ln.startswith("elapsed_sec="):
            try:
                return float(ln.split("=", 1)[1])
            except ValueError:
                return float("nan")
    return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sam-name", required=True)
    p.add_argument("--py-cna", type=Path, required=True)
    p.add_argument("--py-tum", type=Path, required=True)
    p.add_argument("--py-runinfo", type=Path)
    p.add_argument("--r-cna", type=Path, required=True)
    p.add_argument("--r-tum", type=Path, required=True)
    p.add_argument("--r-runinfo", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    py_cna = _read_cna(args.py_cna)
    r_cna = _read_cna(args.r_cna)
    py_tum = _read_lines(args.py_tum)
    r_tum = _read_lines(args.r_tum)

    all_cells = [c for c in r_cna.columns if c not in _ANNOT_COLS]

    row = {"sam_name": args.sam_name}
    row.update(cna_spearman(py_cna, r_cna))
    row.update(tumor_consistency(py_tum, r_tum, all_cells))
    py_sec = _read_elapsed_sec(args.py_runinfo)
    r_sec = _read_elapsed_sec(args.r_runinfo)
    row["py_sec"] = py_sec
    row["r_sec"] = r_sec
    row["speedup"] = speedup(py_sec, r_sec)

    pd.DataFrame([row]).to_csv(args.out_dir / "metrics.csv", index=False)
    print(
        f"[compare] {args.sam_name}: "
        f"CNA rho={row['cna_spearman_overall']:.4f} "
        f"(cell median {row['cna_spearman_cell_median']:.4f}), "
        f"tumour ARI={row['tumor_ari']:.4f} Jaccard={row['tumor_jaccard']:.4f} "
        f"(overlap {row['tumor_overlap']}, py_only {row['tumor_py_only']}, "
        f"r_only {row['tumor_r_only']}), speedup={row['speedup']}"
    )


if __name__ == "__main__":
    main()
