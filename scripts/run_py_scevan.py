"""Run pyscevan ``pipeline_cna`` on one patient and dump benchmark artifacts.

Route A: the confident-normal set is whatever R auto-detected
(``findConfident=TRUE``); we read it from ``norm_cell.txt`` and feed the
*identical* list to py, so the timed work (preprocess + classify + clonal CN)
is the same on both sides and the only thing under test is the reconstruction.

Outputs (uniform with the R driver) into ``--out-dir``::

    <sam>_py_CNAmat.tsv.gz   gene_id/seqnames/end + cells (relativised body)
    <sam>_py_tum_cells.txt   one tumour cell name per line
    <sam>_py_classDf.tsv     index=cell, columns class/confidentNormal
    <sam>_py_runinfo.txt     elapsed_sec=<wall clock of pipeline_cna>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pyscevan import pipeline_cna


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--counts", type=Path, required=True,
                   help="genes x cells raw counts TSV (gene index, cell header)")
    p.add_argument("--norm-cell", type=Path, required=True,
                   help="norm_cell.txt produced by the R driver (route A)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sam-name", required=True)
    p.add_argument("--organism", default="human")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if str(args.counts).endswith(".gz") else None
    count_mtx = pd.read_csv(
        args.counts, sep="\t", index_col=0, compression=compression
    )
    norm_cell = [ln.strip() for ln in open(args.norm_cell) if ln.strip()]
    # keep only normals actually present (R may have filtered some cells out)
    norm_cell = [c for c in norm_cell if c in count_mtx.columns]

    t0 = time.perf_counter()
    res = pipeline_cna(
        count_mtx,
        norm_cell=norm_cell,
        sample=args.sam_name,
        subclones=False,
        clonal_cn=True,
        organism=args.organism,
    )
    elapsed = time.perf_counter() - t0

    res.cna_matrix.to_csv(
        args.out_dir / f"{args.sam_name}_py_CNAmat.tsv.gz",
        sep="\t", index=False, compression="gzip",
    )
    tum = res.class_df.index[res.class_df["class"] == "tumor"].tolist()
    (args.out_dir / f"{args.sam_name}_py_tum_cells.txt").write_text(
        "\n".join(tum) + "\n"
    )
    res.class_df.to_csv(args.out_dir / f"{args.sam_name}_py_classDf.tsv", sep="\t")
    (args.out_dir / f"{args.sam_name}_py_runinfo.txt").write_text(
        f"elapsed_sec={elapsed:.4f}\n"
        f"n_tumor={len(tum)}\n"
        f"n_cells={res.cna_matrix.shape[1] - 3}\n"
    )
    print(
        f"[py] {args.sam_name}: {elapsed:.2f}s, "
        f"{len(tum)} tumour / {res.cna_matrix.shape[1] - 3} classified"
    )


if __name__ == "__main__":
    main()
