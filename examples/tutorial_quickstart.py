"""Quickstart tutorial for pyscevan.

Jupytext percent-format source. Use `examples/_build_notebooks.py` to turn this
into `tutorial_quickstart.ipynb` (and optionally `.executed.ipynb`).

Runs on the bundled MGH106 frozen subset (the same fixture the R-parity tests
use), so it needs no download and finishes in a few seconds. matplotlib-only;
no `omicverse` import.
"""
# ruff: noqa: E402  # jupytext percent-format: imports live inside cells

# %% [markdown]
# # pyscevan — quickstart
#
# [`pyscevan`](https://github.com/omicverse/py-SCEVAN) is a pure-Python rewrite
# of the R tumour-CNA caller [SCEVAN](https://github.com/AntonioDeFalco/SCEVAN).
# This notebook runs the MVP **single-sample, user-known-normal, single-clone**
# path end-to-end on the bundled **MGH106** glioblastoma subset (200 cells) that
# ships with the repo's R-parity fixtures — no download, a few seconds to run.
#
# What the notebook does:
#
# 1. Loads the genes × cells count matrix + the confident-normal cell list from
#    `tests/r_ref/mgh106/`.
# 2. Runs `pyscevan.pipeline_cna(...)` (the DataFrame API) and inspects the
#    returned `SCEVANResult` (per-cell class, CNA matrix, clonal CN profile).
# 3. Repeats the run through the **AnnData-native** path (the I/O contract used
#    by the `pyscevan run-h5ad` CLI) and checks the two agree.
# 4. Draws a chromosome-ordered CNA heatmap (normal vs tumour cells).
#
# > **Scope.** The MVP requires a user-supplied `norm_cell` list; the automatic
# > confident-normal detection (yaGST), subclones, and multi-sample comparison
# > are not ported yet — see `NAMESPACE_PARITY.md`.

# %% [markdown]
# ## Installation
#
# If `pyscevan` is already installed in this environment (it is, under the
# repo's uv venv), the next cell is a no-op.
#
# ```bash
# uv sync               # from the pyscevan repo root
# # or, once released:
# pip install pyscevan
# ```

# %%
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyscevan
from pyscevan import SCEVANResult, pipeline_cna

print("pyscevan version:", getattr(pyscevan, "__version__", "0.1.0.dev0"))

# Repo root is two levels up from the installed package (works under uv venv).
REPO_ROOT = Path(pyscevan.__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "r_ref" / "mgh106"
print("fixture dir:", FIXTURE)

# %% [markdown]
# ## 1. Load the bundled MGH106 fixture
#
# Two files:
#
# | file | contents |
# |---|---|
# | `mgh106_subset.tsv.gz` | genes × cells raw counts (gene symbols on the index, cell barcodes on the header) |
# | `norm_cell.txt` | one confident-normal cell barcode per line (what R auto-detected, frozen for determinism) |
#
# SCEVAN/pyscevan consume a **genes × cells** matrix plus the explicit list of
# confident-normal cells.

# %%
count_mtx = pd.read_csv(
    FIXTURE / "mgh106_subset.tsv.gz", sep="\t", index_col=0, compression="gzip"
)
norm_cell = [ln.strip() for ln in open(FIXTURE / "norm_cell.txt") if ln.strip()]

print("count matrix (genes × cells):", count_mtx.shape)
print("confident-normal cells:", len(norm_cell))
count_mtx.iloc[:4, :3]

# %% [markdown]
# ## 2. Run `pyscevan.pipeline_cna` (DataFrame API)
#
# The single MVP entry point mirrors R `pipelineCNA(SUBCLONES=FALSE)`:
#
# * `norm_cell` — the confident-normal barcodes (required in the MVP).
# * `subclones=False` — single-clone MVP path.
# * `clonal_cn=True` — also infer the clonal copy-number profile.
#
# It returns a `SCEVANResult` dataclass.

# %%
result = pipeline_cna(
    count_mtx,
    norm_cell=norm_cell,
    sample="MGH106s",
    subclones=False,
    clonal_cn=True,
)
assert isinstance(result, SCEVANResult)

# Per-cell class: filtered / normal / tumour.
class_counts = result.class_df["class"].value_counts()
print("per-cell class counts:")
print(class_counts.to_string())
print(f"\nCNA matrix (genes × annotated cells): {result.cna_matrix.shape}")
print(f"clonal CN segments: {len(result.clonal_cn)}")

# %% [markdown]
# `result.class_df` is indexed by cell barcode, with a `class` column
# (`filtered` / `normal` / `tumor`) and a `confidentNormal` flag mirroring R's
# `classDf`. On this fixture it reproduces the R reference exactly: **136 tumour
# / 64 normal**.

# %%
result.class_df.head()

# %% [markdown]
# The clonal CN profile (`getClonalCNProfile` in R) gives a per-segment integer
# copy-number call across the genome for the tumour compartment.

# %%
result.clonal_cn.head(8)

# %% [markdown]
# ## 3. The AnnData-native path
#
# Single-cell tools speak [AnnData](https://anndata.readthedocs.io/). pyscevan's
# `pyscevan run-h5ad` CLI and the `pyscevan.io.anndata_io` adapters bridge a
# **cells × genes** AnnData (with an `obs` column flagging normal cells) to the
# genes × cells + `norm_cell` convention `pipeline_cna` wants.
#
# Here we wrap the same fixture in an AnnData (normals flagged in
# `obs["is_normal"]`) and confirm the AnnData path reproduces the DataFrame run.

# %%
import anndata as ad

from pyscevan.io.anndata_io import adata_to_count_mtx, normal_cells_from_obs

# cells × genes AnnData (transpose of the genes × cells count matrix).
adata = ad.AnnData(
    X=count_mtx.T.to_numpy(dtype=np.float32),
    obs=pd.DataFrame(
        {"is_normal": [c in set(norm_cell) for c in count_mtx.columns]},
        index=count_mtx.columns,
    ),
    var=pd.DataFrame(index=count_mtx.index),
)
print(adata)

# Pull the SCEVAN-shaped inputs back out of the AnnData.
count_mtx_from_adata = adata_to_count_mtx(adata)
norm_from_adata = normal_cells_from_obs(adata, normal_key="is_normal")
print("\nnormal cells recovered from obs:", len(norm_from_adata))

result_adata = pipeline_cna(
    count_mtx_from_adata,
    norm_cell=norm_from_adata,
    sample="MGH106s",
    subclones=False,
    clonal_cn=True,
)

# The two routes are the same computation -> identical class calls.
same = result_adata.class_df["class"].equals(result.class_df["class"])
print("AnnData path matches DataFrame path (per-cell class):", same)

# %% [markdown]
# ## 4. CNA heatmap (chromosome-ordered, normal vs tumour)
#
# `result.cna_matrix` holds the relativised CNA signal: the leading columns are
# `gene_id / seqnames / end` (gene id, chromosome, position) and the remaining
# columns are cells. Genes are already in genomic order, so we just split cells
# into normal/tumour and draw the cells × genes image with chromosome dividers.

# %%
annot_cols = ["gene_id", "seqnames", "end"]
cell_cols = [c for c in result.cna_matrix.columns if c not in annot_cols]
chrom = result.cna_matrix["seqnames"].to_numpy()

# Order cells: normal block first, then tumour, each as a contiguous group.
cls = result.class_df["class"]
normal_cells = [c for c in cell_cols if cls.get(c) == "normal"]
tumor_cells = [c for c in cell_cols if cls.get(c) == "tumor"]
ordered = normal_cells + tumor_cells

# cells × genes matrix, clipped for display.
img = result.cna_matrix[ordered].to_numpy(dtype=float).T
vlim = np.percentile(np.abs(img), 98)

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(img, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
               interpolation="nearest")

# chromosome boundaries + centred labels along the gene (x) axis.
bounds = np.flatnonzero(chrom[1:] != chrom[:-1]) + 1
for b in bounds:
    ax.axvline(b, color="k", lw=0.4, alpha=0.5)
starts = np.concatenate(([0], bounds))
ends = np.concatenate((bounds, [len(chrom)]))
ax.set_xticks((starts + ends) / 2)
ax.set_xticklabels([str(chrom[s]) for s in starts], fontsize=7)
ax.set_xlabel("chromosome")

# horizontal divider between the normal and tumour blocks.
ax.axhline(len(normal_cells) - 0.5, color="k", lw=1.2)
ax.set_yticks([len(normal_cells) / 2, len(normal_cells) + len(tumor_cells) / 2])
ax.set_yticklabels([f"normal\n(n={len(normal_cells)})",
                    f"tumour\n(n={len(tumor_cells)})"], fontsize=9)
ax.set_title("pyscevan MGH106 — relativised CNA (normal vs tumour cells)")
fig.colorbar(im, ax=ax, shrink=0.6, label="relative CNA")
plt.tight_layout()
plt.show()

# %% [markdown]
# The tumour block shows clear chromosome-arm-level gains/losses (red/blue) that
# the normal block lacks — the signal SCEVAN clusters on to call malignancy.
#
# ## Next steps
#
# * **AnnData / CLI** — the same run from the shell:
#   `pyscevan run-h5ad --input data.h5ad --normal-key is_normal --output out/`.
# * **Parity status** — `pyscevan` reproduces R SCEVAN on MGH106 exactly
#   (tum_cells 136/136, CNA matrix to float noise). See `README.md`'s parity
#   table, `NAMESPACE_PARITY.md`, and the tier-4 checks in `tests/`.
# * **Benchmark** — `benchmarks/` runs a py↔R consistency benchmark across 3CA
#   patients (speed / CNA Spearman / tumour ARI-Jaccard); see `benchmarks/README.md`.
#   pyscevan reproduces R to Spearman ρ = 1.0 / ARI = 1.0 at a multi-× speedup.
# * **Not yet ported** — automatic confident-normal detection (yaGST), subclone
#   analysis, multi-sample comparison, plotting/oncoprint (`NAMESPACE_PARITY.md`).
