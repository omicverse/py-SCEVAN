# pyscevan

A pure-Python re-implementation of [SCEVAN](https://github.com/AntonioDeFalco/SCEVAN) (De Falco et al., *Nat Commun* 2023) — tumour-cell calling and clonal copy-number profiling from scRNA-seq counts. Drop-in for the scanpy / AnnData ecosystem.

- **No `rpy2`**, no R install — the single-sample SCEVAN CNA pipeline (preprocess → annotate → vegaMC segmentation → EM CN-call → Ward tumour classification → clonal CN profile) is implemented directly in NumPy / SciPy / scikit-learn / Numba.
- Same low-level function surface as the R workflow (`annotateGenes` / `preprocessingMtx` / `getBreaksVegaMC` / `computeCNAmtx` / `classifyTumorCells` / `getCNcall` / `pipelineCNA`) — see [NAMESPACE_PARITY.md](NAMESPACE_PARITY.md).
- Pure-Python wheel (`py3-none-any`) — no compiler required.

> **MVP scope.** This is the **user-known-normal, single-clone** path of SCEVAN. The auto confident-normal detection (yaGST), subclone analysis, plotting/HTML, multi-sample comparison, and phylogenetic tree are **not yet ported** (see [Known divergences / limitations](#known-divergences--limitations)). Verified end-to-end on the **MGH106** glioblastoma sample.

> This is a **candidate standalone mirror** of the canonical implementation that will live in [`omicverse`](https://github.com/Starlitnightly/omicverse) (`omicverse.external.scevan_py`). Algorithmic work is developed here first and synced upstream for users who want SCEVAN without the full omicverse stack.

## Install

```bash
pip install pyscevan
```

or, from source with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Python 3.10+ supported. Pure-Python wheel — no compiler required.

## Quick-start

```python
import pandas as pd
from pyscevan import pipeline_cna

count_mtx = pd.read_csv("counts.tsv", sep="\t", index_col=0)  # genes × cells, raw ints

result = pipeline_cna(
    count_mtx,
    norm_cell=["cell_017", "cell_042", ...],   # confident-normal cell names (MVP: required)
    sample="MGH106",
    organism="human",
)

result.class_df     # DataFrame[cell -> class (filtered/normal/tumor) + confidentNormal]
result.cna_matrix   # CNAmat: gene_id/seqnames/end + per-cell relativised CNA body
result.clonal_cn    # clonal CN profile: Chr/Pos/End/CN (None when ClonalCN=False)
```

## Low-level functional API (mirrors R one-to-one)

```python
from pyscevan.io.annotation import annotate_genes
from pyscevan.preprocess import preprocessing_mtx
from pyscevan.segment.vegamc import get_breaks_vegamc
from pyscevan.cna import compute_cna_mtx
from pyscevan.classify.tumor import classify_tumor_cells
from pyscevan.classify.cncall import get_cn_call
```

See [NAMESPACE_PARITY.md](NAMESPACE_PARITY.md) for the full R-export → Python symbol map.

## Parity status

These numbers are from the **gated R-parity tests** (`tests/`), measured against
R SCEVAN run on the **MGH106** sample. They are reported verbatim — `bit-exact`,
`exact`, and the per-component `atol` figures are the actual measured values, not
optimistic rounding. **NON-parity** items are flagged explicitly.

| Component (R → Python) | Parity | Evidence |
|---|---|---|
| `getBreaksVegaMC` breakpoints (Chr / Start / End / Size / Probe Size) | **bit-exact** | c1–c9 `array_equal` |
| `getBreaksVegaMC` segment `Mean` | value-parity, atol ~5e-7 | float32 reduction order (R writes `%f`) |
| `getBreaksVegaMC` loss/gain % (`X..L` / `X.G`) + `Loss/Gain Mean` | value-parity (exact % strings; Mean atol 1e-4) | c1–c9 |
| `getBreaksVegaMC` bootstrap p-values (`L.pv` / `G.pv`) | **NON-parity** | C libc `rand()` vs `np.random`; recipe (quantize / `<=` / 5-dp round) matched, but the RNG stream differs → soft-tested only |
| `annotateGenes` | **identity** (exact gene set + order) | 16570 genes, MGH106 |
| `preprocessingMtx` (`count_mtx_norm`) | value-parity **7.1e-15** (float64, ~bit-exact) | MGH106 10566 × 200 |
| EM CN-call (CN labels) | **exact** 60 / 60 | MGH106 clonal |
| `computeCNAmtx` | value-parity **7.77e-16** | |
| `classifyTumorCells` `tum_cells` | **exact** 136 / 136 | ward.D via sqrt-trick |
| `classifyTumorCells` `CNAmat` | value-parity **5.77e-15** | |
| `pipelineCNA` `classDf` (`class` + `confidentNormal`) | **exact** (136 tumor / 64 normal) | MGH106 |
| `pipelineCNA` clonal CN (Chr / Pos / End / CN) | **exact** 60 segs | |
| `nonlinear_smooth` (tanh) | **bit-exact** (0.0) | |

### Generating the R reference

```bash
Rscript scripts/gen_r_reference.R          # writes tests/r_ref/*.tsv
uv run pytest tests/test_r_parity.py -v
```

The Python-side tests `pytest.skip` cleanly when the reference TSVs are absent
(they do not fail).

## Known divergences / limitations

Honest, exhaustive list. The first two are the load-bearing caveats; the rest
are measure-zero or cosmetic.

- **vegaMC bootstrap p-values are NON-parity.** SCEVAN's C `vegaMC` kernel draws
  bootstrap samples with libc `rand()`, which is not reproducible across
  platforms / compilers. pyscevan reproduces the *recipe* (quantization, `<=`
  comparison, 5-decimal rounding) but its `np.random` stream is a different
  sequence. Only `abs(Mean) > 0.05` and the soft p-value *direction* are tested;
  the exact `L.pv` / `G.pv` values are **not** asserted equal.
- **`classify` `segmAlt` uses the non-parity p-value term.** A segment is flagged
  altered by `abs(Mean) > 0.05` **OR** the bootstrap p-value. On MGH106,
  124 / 150 segments are flagged by the deterministic `abs(Mean) > 0.05` term and
  **4 / 150 are flagged ONLY by the p-value**. On this sample those 4 did not flip
  the tumour/normal result, but a different sample could diverge — so the test
  carries an **ARI > 0.95 floor fallback** rather than asserting bit-exact `segmAlt`.
- **ward.D linkage via the `scipy ward(sqrt(d))` equivalence.** R `hclust(..., method="ward.D")`
  is reproduced using the identity `scipy 'ward' == R 'ward.D2'` applied to
  `sqrt(d)`. On MGH106 the 2-cut partition coincides with `ward.D2`, so this
  dataset can only show the trick **correct / consistent**, not **necessary**.
  Linkage *heights* differ (monotone transform); the *partition* matches exactly.
- **`classifyCluster` exact-tie.** On an exact tie Python picks cluster 1; R's
  `which(== max)` has undefined recycling behaviour on exact ties (measure-zero,
  continuous inputs).
- **EM `getLabelCall` tie-break.** R uses `max.col(ties="random")`; pyscevan uses
  `np.argmax` (first). Measure-zero divergence for continuous posteriors.
- **`classDf` "filtered" class.** R lists cells dropped by the per-chromosome
  filter as `"filtered"`, indexed against the *pre-per-chr-filter* count matrix.
  pyscevan's `classDf` covers the **analyzed** cells. On MGH106 the per-chr filter
  drops 0 cells, so the two coincide exactly here; for samples that *do* drop
  cells per-chromosome the "filtered" rows would differ. Documented gap.
- **Clonal CN omits R's cosmetic `segm.mean` column.** R `getClonalCNProfile`
  `cbind`s a `segm.mean` column read back from the beta=3 vega output;
  pyscevan returns the raw `getCNcall` output (Chr / Pos / End / CN) without it.

### Not ported (Phase 1.5 / Phase 2)

- **Auto confident-normal detection (yaGST).** `find_confident=True` /
  `norm_cell=None` raise — the MVP requires user-provided normal cells.
- **Subclone analysis.** `SUBCLONES=True` is accepted but **ignored with a
  warning** (single-clone MVP).
- **Plotting / HTML, multi-sample comparison, phylogenetic tree.** Not ported.

### py3-none-any note

pyscevan's **own** wheel has no compiled extensions — it is pure bytecode and
ships the `py3-none-any` tag. Its runtime dependencies
(`numpy` / `scipy` / `scikit-learn` / `numba` / `anndata`) ship their own binary
wheels from PyPI; that is normal and does **not** violate the pure-Python-wheel
target for pyscevan itself.

## Testing

```bash
uv run pytest tests/unit tests/test_smoke.py    # fast unit + smoke tests (no R)
uv run pytest tests/test_r_parity.py            # offline R-parity (skips without tests/r_ref/)
```

## Relationship to omicverse

Developed following the [omicverse-to-developer](https://github.com/omicverse/omicverse-to-developer)
`py-<Name>` conventions (pure-Python, no `rpy2` in production code, AnnData-native
I/O, Numba only on hot kernels). Upstream integration plan:

- Canonical implementation: `omicverse.external.scevan_py` (pending)
- Standalone mirror (this repo): same code, same API, without the full omicverse packaging

## Citation

If you use this package, please cite the original SCEVAN paper:

> De Falco A, Caruso F, Su X-D, Iavarone A, Ceccarelli M. A variational algorithm to detect the clonal copy number substructure of tumours from single-cell data. Nat Commun. 2023;14(1):1074. doi:10.1038/s41467-023-36790-9

and acknowledge omicverse / this repository for the Python port.

## License

GPL-2.0-or-later — derivative of GPL-2 SCEVAN by Antonio De Falco.
