# pyscevan benchmarks — py↔R consistency on 3CA

Mirrors the pycopykat 17-patient benchmark paradigm. The target is **how
faithfully the Python reconstruction reproduces R SCEVAN**, across four
dimensions Jason specified:

1. **Speed** — `speedup = R wall-clock / py wall-clock` on the ported region.
2. **CNA matrix** — Spearman ρ, py `cna_matrix` vs R `CNAmat` (overall + per-cell).
3. **Functional coverage** — static, from [`../NAMESPACE_PARITY.md`](../NAMESPACE_PARITY.md)
   (4/11 R exports ported + 5 internal helpers; 7/11 deferred).
4. **Tumour calls** — ARI / Jaccard, py `tum_cells` vs R `tum_cells`.

> **No external truth is used.** The 3CA `cell_type` column (including its own
> `Malignant` call) is *itself* an algorithm output, so it is not a reference.
> Every metric measures py↔R agreement only.

## Route A — matched normals

pyscevan's MVP requires a user-supplied `norm_cell` (the yaGST auto-normal path
is unported, Phase 1.5). So `run_r_scevan.R` lets R auto-detect its confident
normals (`findConfident=TRUE`), **freezes** them to `norm_cell.txt`, and feeds
the identical set to py. Both pipelines therefore run on the same normals; the
auto-detect cost is excluded from the timed region on both sides.

## Data

The 17 3CA patients are the frozen slices already living in the sibling
`pycopykat/benchmarks/full/<dataset>/<patient>/` (`counts.tsv` + `cells.csv`).
Per workspace convention they are *referenced*, not copied — see `--data-root`
in `benchmark_config.py` (default `../pycopykat/benchmarks/full`).

## Reproducing

**R side** (on `<r-host>`, SCEVAN in `~/Rlib`, long → tmux). Per patient:

```bash
Rscript scripts/run_r_scevan.R <counts.tsv> <out_dir> <label__patient>
# writes: norm_cell.txt, <sam>_r_CNAmat.tsv.gz, <sam>_r_tum_cells.txt,
#         <sam>_r_classDf.tsv, <sam>_r_runinfo.txt
```

Pull each patient's R artifacts into
`benchmarks/full/<label>/<patient>/r_out/`.

**py side + comparison** (local, uv):

```bash
uv run python scripts/run_py_sweep.py        # py pipeline_cna + compare per patient
uv run python scripts/aggregate_benchmark.py # -> benchmarks/summary.csv + SUMMARY.md
uv run --with matplotlib python scripts/make_overview_figure.py
```

## Framework validation (MGH106)

`benchmarks/mgh106/` validates the harness end-to-end against the frozen
`tests/r_ref/mgh106` reference (no server needed): CNA ρ = 1.0000, tumour
ARI = 1.0000, Jaccard = 1.0000 (136/136). `speedup` is `n/a` there (the frozen
subset carries no R wall-clock).

## Files

| Script | Role |
|---|---|
| `run_r_scevan.R` | R reference per patient (route A), server-side |
| `run_py_scevan.py` | pyscevan `pipeline_cna` per patient, timed |
| `compare_py_vs_r.py` | per-patient 4-dimension `metrics.csv` |
| `bench_metrics.py` | metric definitions (Spearman / ARI / Jaccard / speedup) |
| `run_py_sweep.py` | local orchestrator (py + compare over all patients) |
| `aggregate_benchmark.py` | `summary.csv` + `SUMMARY.md` |
| `make_overview_figure.py` | 3-panel overview PNG |
| `benchmark_config.py` | 17-patient manifest + data-root resolver |
