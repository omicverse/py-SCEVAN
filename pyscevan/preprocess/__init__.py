"""preprocessing_mtx — port of SCEVAN preprocessingMtx (preProcessing.R:74-185).

Filters cells/genes, annotates genomic coordinates, coordinate-sorts, removes
cell-cycle + HLA genes, applies a per-chromosome gene-count filter, then a
Log-Freeman-Tukey transform with per-cell centering. float64 throughout (the
float32 traps live only in the vegaMC kernel, not here).

MVP scope: ``find_confident=False`` skips getConfidentNormalCells (the
user-normal path), so ``count_mtx_norm`` is yaGST-independent.
"""

from importlib.resources import files
from typing import NamedTuple

import numpy as np
import pandas as pd

from ..io.annotation import annotate_genes

_D = files("pyscevan.data")


class PreprocessResult(NamedTuple):
    count_mtx_norm: pd.DataFrame   # genes x cells, float64, index = gene_name
    count_mtx_annot: pd.DataFrame  # 5 annotation cols, coordinate-sorted


def _load_cellcycle(organism: str = "human") -> set[str]:
    """reactome_cellcycle gene symbols, one per line (mirror sysdata.py loaders)."""
    fn = (
        "reactome_cellcycle.txt"
        if organism == "human"
        else "reactome_cellcycle_Mmusculus.txt"
    )
    text = (_D / fn).read_text()
    return {ln.strip() for ln in text.splitlines() if ln.strip()}


def preprocessing_mtx(
    count_mtx: pd.DataFrame,
    sample: str = "",  # noqa: N803 — R kwarg fidelity
    ngenes_chr: int = 5,  # noqa: N803
    perc_genes: float = 0.1,  # noqa: N803
    find_confident: bool = False,  # noqa: N803
    organism: str = "human",
) -> PreprocessResult:
    """Port of SCEVAN::preprocessingMtx (preProcessing.R:74-185).

    Parameters
    ----------
    count_mtx
        genes x cells count matrix (symbols or Ensembl ids on the index).
    sample
        Sample name (unused on the find_confident=False path, kept for parity).
    ngenes_chr
        Minimum run-length of consecutive same-chromosome genes per cell.
    perc_genes
        FRACTION (R semantics): keep genes expressed in > perc_genes of cells.
        NOTE: SCEVANConfig.perc_genes is 10.0 (a percentage); reconcile in T10.
    find_confident
        MVP: must be False (skips getConfidentNormalCells / yaGST).
    organism
        "human" (totChr=22) or mouse (totChr=19).

    Returns
    -------
    PreprocessResult(count_mtx_norm, count_mtx_annot).
    """
    # preProcessing.R:82-89 — 1) filter cells > 200 genes (exact guard).
    # genes.raw = colSums(count_mtx > 0). Removal happens ONLY inside the
    # `if sum(genes.raw < 100) > 1` guard, dropping cells with genes.raw < 200.
    genes_raw = (count_mtx > 0).sum(axis=0)
    if (genes_raw > 200).sum() == 0:
        raise ValueError("none cells have more than 200 genes")
    if (genes_raw < 100).sum() > 1:
        keep_cells = ~(genes_raw < 200)
        count_mtx = count_mtx.loc[:, keep_cells]

    # preProcessing.R:91-101 — 2) filter genes by fraction of cells expressing.
    ncol = count_mtx.shape[1]
    der = (count_mtx > 0).sum(axis=1) / ncol
    if (der > perc_genes).sum() > 7000:
        count_mtx = count_mtx.loc[der > perc_genes]
    else:
        perc_genes = perc_genes - 0.05
        count_mtx = count_mtx.loc[der > perc_genes]

    # preProcessing.R:108-111 — 3) annotate coordinates; then count_mtx becomes
    # the non-annotation columns, rownames = gene_name (handled by annot order).
    count_mtx_annot = annotate_genes(count_mtx, organism)

    # find_confident=False -> norm_cell = NULL (MVP user-normal path). skip.

    # preProcessing.R:121-123 — 4) coordinate sort by (seqnames, end) ascending.
    # seqnames/end already numeric (annotate_genes coerces seqnames; end numeric).
    count_mtx_annot = count_mtx_annot.sort_values(
        by=["seqnames", "end"], kind="stable"
    ).reset_index(drop=True)

    # preProcessing.R:129-141 — 5) remove cell-cycle + HLA genes.
    cellcycle = _load_cellcycle(organism)
    gene_name = count_mtx_annot["gene_name"]
    hlas = set(gene_name[gene_name.str.match("^HLA-")])
    to_rev = gene_name.isin(cellcycle | hlas)
    if to_rev.any():
        count_mtx_annot = count_mtx_annot[~to_rev].reset_index(drop=True)

    total_chr = 22 if organism == "human" else 19

    # preProcessing.R:145-161 — 6) per-chromosome gene filter.
    # For each cell: take seqnames of genes with expr != 0 (genes are sorted by
    # seqnames, so runs are contiguous). rle: #distinct runs == #chromosomes.
    # Drop cell if (#chromosomes < total_chr) OR (min run-length < ngenes_chr).
    seqnames_arr = count_mtx_annot["seqnames"].to_numpy()
    cell_cols = list(count_mtx_annot.columns[5:])
    expr = count_mtx_annot[cell_cols].to_numpy()
    cells_filt = []
    for j, col in enumerate(cell_cols):
        nz = expr[:, j] != 0
        seq = seqnames_arr[nz]
        if seq.size == 0:
            cells_filt.append(col)
            continue
        run_lengths = _rle_lengths(seq)
        if len(run_lengths) < total_chr or run_lengths.min() < ngenes_chr:
            cells_filt.append(col)

    if len(cells_filt) == len(cell_cols):
        raise ValueError("all cells are filtered")
    if cells_filt:
        count_mtx_annot = count_mtx_annot.drop(columns=cells_filt)

    if (count_mtx_annot.shape[1] - 5) < 15:
        raise ValueError("Bad sample low cells < 15")

    # preProcessing.R:167-176 — 7) Log-Freeman-Tukey + 8) column-center.
    cell_cols = list(count_mtx_annot.columns[5:])
    proc = count_mtx_annot[cell_cols].to_numpy(dtype=np.float64)
    annot5 = count_mtx_annot.iloc[:, :5].reset_index(drop=True)
    norm = np.log(np.sqrt(proc) + np.sqrt(proc + 1.0))  # outer log REQUIRED
    norm = norm - norm.mean(axis=0, keepdims=True)      # per-cell (column) center

    count_mtx_norm = pd.DataFrame(
        norm, index=annot5["gene_name"].to_numpy(), columns=cell_cols
    )
    count_mtx_norm.index.name = None

    return PreprocessResult(count_mtx_norm=count_mtx_norm, count_mtx_annot=annot5)


def _rle_lengths(x: np.ndarray) -> np.ndarray:
    """Run-length encoding lengths of a 1-D array (R rle()$lengths)."""
    if x.size == 0:
        return np.array([], dtype=int)
    change = np.nonzero(np.diff(x))[0] + 1
    bounds = np.concatenate(([0], change, [x.size]))
    return np.diff(bounds)
