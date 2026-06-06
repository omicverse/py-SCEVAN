"""Overview figure for the pyscevan vs R SCEVAN benchmark (3 panels).

Reads ``benchmarks/summary.csv`` (from aggregate_benchmark.py) and renders:
    1. speedup per patient (bar)
    2. CNA Spearman rho per patient (overall + per-cell median)
    3. tumour-call agreement per patient (ARI + Jaccard)

matplotlib is benchmark-only and NOT a package dependency. Run with::

    uv run --with matplotlib python scripts/make_overview_figure.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark_config import PROJECT  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", type=Path, default=PROJECT / "benchmarks")
    args = p.parse_args()

    df = pd.read_csv(args.bench_dir / "summary.csv").sort_values("sam_name")
    n = len(df)
    x = np.arange(n)
    labels = df["sam_name"].tolist()

    fig, axes = plt.subplots(3, 1, figsize=(max(8, n * 0.55), 11), sharex=True)

    # 1. speedup
    sp = df["speedup"].to_numpy(dtype=float)
    axes[0].bar(x, np.where(np.isfinite(sp), sp, 0.0), color="#4C72B0")
    finite_sp = sp[np.isfinite(sp)]
    if finite_sp.size:
        m = float(np.nanmean(sp))
        axes[0].axhline(m, color="#C0392B", ls="--", lw=1,
                        label=f"mean {m:.1f}×")
        axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("speedup (R / py)")
    axes[0].set_title("pyscevan vs R SCEVAN — py↔R consistency benchmark", fontsize=12)

    # 2. CNA Spearman
    axes[1].plot(x, df["cna_spearman_overall"], "o-", color="#55A868",
                 label="overall ρ")
    axes[1].plot(x, df["cna_spearman_cell_median"], "s--", color="#8172B3",
                 label="per-cell median ρ")
    axes[1].set_ylabel("CNA Spearman ρ")
    axes[1].set_ylim(min(0.8, float(df["cna_spearman_cell_median"].min()) - 0.02), 1.005)
    axes[1].legend(frameon=False, fontsize=8)

    # 3. tumour agreement
    axes[2].plot(x, df["tumor_ari"], "o-", color="#DD8452", label="ARI")
    axes[2].plot(x, df["tumor_jaccard"], "s--", color="#937860", label="Jaccard")
    axes[2].set_ylabel("tumour-call agreement")
    axes[2].set_ylim(min(0.8, float(df["tumor_jaccard"].min()) - 0.02), 1.005)
    axes[2].legend(frameon=False, fontsize=8)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    for ax in axes:
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    fig.tight_layout()
    out = args.bench_dir / "overview_figure.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[figure] wrote {out}")


if __name__ == "__main__":
    main()
