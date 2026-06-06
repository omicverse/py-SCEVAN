# NAMESPACE parity — R SCEVAN → pyscevan

Audit of the new-port checklist item:
> NAMESPACE 每个 R export 在 `__all__` 或 submodule 有 Python 等价物。

Upstream `SCEVAN/NAMESPACE` exports **11** names. Cross-reference below. The MVP
ports the **single-sample, user-known-normal, single-clone** path; auto
confident-normal detection (yaGST), subclones, plotting/HTML, and multi-sample
comparison are deferred (see status column).

## Exported R functions

| # | R export | R source | Python equivalent | Path | Status |
|---|---|---|---|---|---|
| 1 | `pipelineCNA` | `pipelineCNA.R:37` | `pipeline_cna` | `pyscevan.pipeline:51` (top-level `__all__`) | **ported (MVP)** — user-normal, single-clone path |
| 2 | `annotateGenes` | `preProcessing.R:10` | `annotate_genes` | `pyscevan.io.annotation:13` | **ported** (submodule) |
| 3 | `getBreaksVegaMC` | `vegaMC.R:119` | `get_breaks_vegamc` | `pyscevan.segment.vegamc:347` | **ported** (submodule) — breakpoints bit-exact; bootstrap p-values NON-parity (see README) |
| 4 | `classifyTumorCells` | `classifyTumor.R:170` | `classify_tumor_cells` | `pyscevan.classify.tumor:115` | **ported** (submodule) |
| 5 | `getConfidentNormalCells` | `confidentNormal.R:61` | — | — | **NOT ported (Phase 1.5)** — yaGST auto-normal detection; MVP requires user-provided `norm_cell` |
| 6 | `top30classification` | `confidentNormal.R:18` | — | — | **NOT ported (Phase 1.5)** — internal helper of `getConfidentNormalCells` (yaGST scoring) |
| 7 | `multiSampleComparisonClonalCN` | `multiSampleComparisonClonalCN.R:79` | — | — | **NOT ported (Phase 2)** — multi-sample comparison |
| 8 | `plotAllClonalCN` | `multiSampleComparisonClonalCN.R:43` | — | — | **NOT ported (Phase 2)** — plotting |
| 9 | `plotAllSubclonalCN` | `multiSampleComparisonClonalCN.R:12` | — | — | **NOT ported (Phase 2)** — subclone plotting |
| 10 | `plotCNA_withAnnotCells` | `plotHeatmap.R:1248` | — | — | **NOT ported (Phase 2)** — heatmap plotting |
| 11 | `annoteBandOncoHeat` | `testFunc.R:921` | — | — | **NOT ported (Phase 2)** — cytoband/oncoprint annotation helper |

## Internal R functions ported (not in NAMESPACE, but algorithmically central)

These are non-exported R functions that the MVP pipeline depends on; they have
first-class Python homes so the low-level path is reproducible.

| R function | R source | Python equivalent | Path |
|---|---|---|---|
| `preprocessingMtx` | `preProcessing.R:74` | `preprocessing_mtx` | `pyscevan.preprocess:39` |
| `computeCNAmtx` | `classifyTumor.R:47` | `compute_cna_mtx` | `pyscevan.cna:16` |
| `getCNcall` | `CNcall.R:283` | `get_cn_call` | `pyscevan.classify.cncall:323` |
| `getClonalCNProfile` | `pipelineCNA.R:91` | (inlined in `pipeline_cna`, step 4) | `pyscevan.pipeline:147` |
| nonlinear `smooth` (tanh) | `classifyTumor.R` (SMOOTH) | `nonlinear_smooth` | `pyscevan.classify.smooth:35` |

## `pyscevan.__all__`

```python
__all__ = ["SCEVANConfig", "SCEVANResult", "pipeline_cna"]
```

`pipeline_cna` is the single top-level entry (mirrors `pipelineCNA`). The
low-level R exports (`annotateGenes`, `getBreaksVegaMC`, `classifyTumorCells`)
live in their organising submodules (`io` / `segment` / `classify`) rather than
being flattened into `__all__`, preserving the `io / preprocess / segment / cna /
classify` structure. `SCEVANConfig` / `SCEVANResult` are pyscevan-original
`@dataclass` config / result containers (no R NAMESPACE equivalent).

## Summary

- **3 / 11 R exports ported** to the low-level surface (`annotateGenes`,
  `getBreaksVegaMC`, `classifyTumorCells`) plus **1 top-level pipeline**
  (`pipelineCNA` → `pipeline_cna`, MVP path).
- **7 / 11 NOT ported**: 2 are the yaGST auto-normal path (Phase 1.5), 5 are
  Phase-2 multi-sample / subclone / plotting / oncoprint exports.
- The MVP additionally ports 5 algorithmically central **internal** R functions
  (`preprocessingMtx`, `computeCNAmtx`, `getCNcall`, `getClonalCNProfile`,
  nonlinear `smooth`) that are not in the R NAMESPACE but are required for the
  end-to-end CNA path.
