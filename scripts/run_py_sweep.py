"""Local py sweep: run pyscevan + compare vs R for every patient with R output.

Prereq: the R side (``run_r_scevan.R``, route A) has already run on the server
and its artifacts have been pulled into
``benchmarks/full/<label>/<patient>/r_out/``:
    norm_cell.txt, <sam>_r_CNAmat.tsv.gz, <sam>_r_tum_cells.txt,
    <sam>_r_runinfo.txt

For each such patient this runs ``run_py_scevan.py`` (feeding R's norm_cell) and
``compare_py_vs_r.py``, writing ``py_out/`` + ``metrics.csv`` next to ``r_out/``.
Resumable: patients whose ``metrics.csv`` already exists are skipped.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from benchmark_config import PROJECT, DEFAULT_DATA_ROOT, DATASETS

VENV_PY = sys.executable
SCRIPTS = PROJECT / "scripts"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run(cmd: list[str]) -> int:
    _log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.call(cmd)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                   help="3CA sliced patients (counts.tsv/cells.csv per patient)")
    p.add_argument("--out-base", type=Path,
                   default=PROJECT / "benchmarks" / "full")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    n_done = n_skip = 0
    for label, patients in DATASETS:
        for pt in patients:
            sam = f"{label}__{pt}"
            pt_out = args.out_base / label / pt
            r_out = pt_out / "r_out"
            py_out = pt_out / "py_out"
            norm_cell = r_out / "norm_cell.txt"
            r_cna = r_out / f"{sam}_r_CNAmat.tsv.gz"
            counts = args.data_root / label / pt / "counts.tsv"

            if not (norm_cell.exists() and r_cna.exists()):
                _log(f"[{sam}] no R output yet -> skip")
                n_skip += 1
                continue
            if (pt_out / "metrics.csv").exists() and not args.force:
                _log(f"[{sam}] metrics.csv present -> skip")
                continue
            if not counts.exists():
                _log(f"[{sam}] !! missing counts {counts} -> skip")
                n_skip += 1
                continue

            rc = _run([VENV_PY, str(SCRIPTS / "run_py_scevan.py"),
                       "--counts", str(counts), "--norm-cell", str(norm_cell),
                       "--out-dir", str(py_out), "--sam-name", sam])
            if rc != 0:
                _log(f"[{sam}] !! py run failed rc={rc}")
                continue
            rc = _run([VENV_PY, str(SCRIPTS / "compare_py_vs_r.py"),
                       "--sam-name", sam,
                       "--py-cna", str(py_out / f"{sam}_py_CNAmat.tsv.gz"),
                       "--py-tum", str(py_out / f"{sam}_py_tum_cells.txt"),
                       "--py-runinfo", str(py_out / f"{sam}_py_runinfo.txt"),
                       "--r-cna", str(r_cna),
                       "--r-tum", str(r_out / f"{sam}_r_tum_cells.txt"),
                       "--r-runinfo", str(r_out / f"{sam}_r_runinfo.txt"),
                       "--out-dir", str(pt_out)])
            if rc == 0:
                n_done += 1
    _log(f"sweep done: {n_done} compared, {n_skip} skipped (no R output)")
    _log("next: scripts/aggregate_benchmark.py")


if __name__ == "__main__":
    main()
