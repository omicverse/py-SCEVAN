"""Aggregate per-patient metrics.csv into a benchmark summary (4 dimensions).

Collects every ``benchmarks/**/metrics.csv`` (3CA patients + optional MGH106),
then writes:
  * ``summary.csv``  -- one row per patient, all metrics.
  * ``SUMMARY.md``   -- a human-readable table + the four-dimension roll-up.

The four dimensions Jason asked for:
  1. speed         -> speedup (R/py wall clock)
  2. CNA matrix    -> Spearman rho (py CNAmat vs R CNAmat)
  3. coverage      -> static, from NAMESPACE_PARITY.md (4/11 ported)
  4. tumour calls  -> ARI / Jaccard (py vs R), py<->R consistency only
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_config import PROJECT

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices ok
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: keep ×/↔

# Dimension 3 -- static coverage, source of truth = NAMESPACE_PARITY.md.
COVERAGE = {
    "r_exports_total": 11,
    "r_exports_ported": 4,        # pipelineCNA + annotateGenes + getBreaksVegaMC + classifyTumorCells
    "r_exports_not_ported": 7,    # yaGST auto-normal (2) + Phase-2 multi/subclone/plot/oncoprint (5)
    "internal_helpers_ported": 5,  # preprocessingMtx, computeCNAmtx, getCNcall, getClonalCNProfile, smooth
}


def _collect(roots: list[Path]) -> pd.DataFrame:
    frames = []
    for root in roots:
        for mpath in sorted(root.rglob("metrics.csv")):
            frames.append(pd.read_csv(mpath))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fmt(x: float, nd: int = 4) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", type=Path, default=PROJECT / "benchmarks")
    args = p.parse_args()

    df = _collect([args.bench_dir])
    if df.empty:
        print(f"[aggregate] no metrics.csv under {args.bench_dir}")
        return
    df = df.sort_values("sam_name").reset_index(drop=True)
    df.to_csv(args.bench_dir / "summary.csv", index=False)

    sp = df["speedup"].to_numpy(dtype=float)
    rho = df["cna_spearman_overall"].to_numpy(dtype=float)
    rho_cell = df["cna_spearman_cell_median"].to_numpy(dtype=float)
    ari = df["tumor_ari"].to_numpy(dtype=float)
    jac = df["tumor_jaccard"].to_numpy(dtype=float)

    lines = [
        "# pyscevan vs R SCEVAN — benchmark summary",
        "",
        f"Patients: **{len(df)}**  ·  py↔R consistency only (no external truth; "
        "the 3CA `Malignant` label is itself an algorithm output, not a reference).",
        "",
        "## Four dimensions",
        "",
        "| Dimension | Metric | Mean | Median | Min |",
        "|---|---|---|---|---|",
        f"| 1. Speed | speedup (R/py) | {_fmt(np.nanmean(sp), 2)}× | "
        f"{_fmt(np.nanmedian(sp), 2)}× | {_fmt(np.nanmin(sp), 2)}× |",
        f"| 2. CNA matrix | Spearman ρ (overall) | {_fmt(np.nanmean(rho))} | "
        f"{_fmt(np.nanmedian(rho))} | {_fmt(np.nanmin(rho))} |",
        f"| 2. CNA matrix | Spearman ρ (per-cell median) | "
        f"{_fmt(np.nanmean(rho_cell))} | {_fmt(np.nanmedian(rho_cell))} | "
        f"{_fmt(np.nanmin(rho_cell))} |",
        f"| 4. Tumour calls | ARI (py vs R) | {_fmt(np.nanmean(ari))} | "
        f"{_fmt(np.nanmedian(ari))} | {_fmt(np.nanmin(ari))} |",
        f"| 4. Tumour calls | Jaccard (py vs R) | {_fmt(np.nanmean(jac))} | "
        f"{_fmt(np.nanmedian(jac))} | {_fmt(np.nanmin(jac))} |",
        "",
        "**3. Functional coverage** (static, see `NAMESPACE_PARITY.md`): "
        f"{COVERAGE['r_exports_ported']}/{COVERAGE['r_exports_total']} R exports "
        f"ported (+{COVERAGE['internal_helpers_ported']} internal helpers); "
        f"{COVERAGE['r_exports_not_ported']}/{COVERAGE['r_exports_total']} not "
        "ported (yaGST auto-normal = Phase 1.5; multi-sample / subclone / "
        "plotting / oncoprint = Phase 2).",
        "",
        "## Per-patient",
        "",
        "| Patient | speedup | CNA ρ | ρ cell-median | tumour ARI | Jaccard | "
        "overlap | py-only | r-only | py s | r s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['sam_name']} | {_fmt(r['speedup'], 2)}× | "
            f"{_fmt(r['cna_spearman_overall'])} | "
            f"{_fmt(r['cna_spearman_cell_median'])} | {_fmt(r['tumor_ari'])} | "
            f"{_fmt(r['tumor_jaccard'])} | {int(r['tumor_overlap'])} | "
            f"{int(r['tumor_py_only'])} | {int(r['tumor_r_only'])} | "
            f"{_fmt(r['py_sec'], 1)} | {_fmt(r['r_sec'], 1)} |"
        )
    (args.bench_dir / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[aggregate] {len(df)} patients -> {args.bench_dir / 'summary.csv'} + SUMMARY.md")
    print("\n".join(lines[5:14]))


if __name__ == "__main__":
    main()
