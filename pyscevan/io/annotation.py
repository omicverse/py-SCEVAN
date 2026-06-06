"""annotate_genes — port of SCEVAN annotateGenes (preProcessing.R:10-51).

Annotates a genes x cells count matrix with genomic coordinates from the
bundled EnsDb (hg38 for human). Pure Python; no coordinate sort here (that
happens in preprocessing_mtx / T6).
"""

import pandas as pd

from .sysdata import load_annotation


def annotate_genes(count_mtx: pd.DataFrame, organism: str = "human") -> pd.DataFrame:
    """Port of SCEVAN::annotateGenes (preProcessing.R:10-51).

    Parameters
    ----------
    count_mtx
        Count matrix with genes on the row index (Ensembl gene_id or Symbol),
        cells on columns.
    organism
        "human" (EnsDb.Hsapiens.v86) or otherwise mouse (EnsDb.Mmusculus.v79).

    Returns
    -------
    DataFrame with 5 annotation columns [seqnames, start, end, gene_id,
    gene_name] column-concatenated with the (subset/reordered) count matrix.
    Row order follows the input count_mtx row order (edb reordered to match).
    No coordinate sort.
    """
    # preProcessing.R:13-17 — pick organism EnsDb.
    edb = load_annotation(organism)

    # preProcessing.R:19-21 — keep seqnames in 1..22, cols [1,2,3,6,7]
    # (seqnames, start, end, gene_id, gene_name); seqnames -> numeric.
    chr_keep = {str(i) for i in range(1, 23)}
    edb = edb[edb["seqnames"].astype(str).isin(chr_keep)]
    edb = edb[["seqnames", "start", "end", "gene_id", "gene_name"]].copy()
    edb["seqnames"] = pd.to_numeric(edb["seqnames"])
    edb = edb.reset_index(drop=True)

    # preProcessing.R:23-27 — branch: any rowname matches a gene_name -> use
    # gene_name (symbol matrix), else gene_id.
    rownames = count_mtx.index
    if rownames.isin(edb["gene_name"]).any():
        use_geneID = "gene_name"  # noqa: N806
    else:
        use_geneID = "gene_id"  # noqa: N806

    # preProcessing.R:29-31 — intersect; subset mtx rows and edb rows.
    # R intersect() keeps the order of the first argument (rownames(mtx)),
    # but mtx is subset via `which(rownames(mtx) %in% genes_inters)` which
    # preserves mtx order; edb is subset via `%in%` preserving edb order.
    inters = set(rownames) & set(edb[use_geneID])
    mtx = count_mtx.loc[rownames.isin(inters)]
    edb = edb[edb[use_geneID].isin(inters)]

    if use_geneID == "gene_id":
        # preProcessing.R:33-37 — reorder edb to mtx order, then dedup
        # gene_name on BOTH edb and mtx (indices aligned).
        order = _order_match(edb[use_geneID].to_numpy(), mtx.index.to_numpy())
        edb = edb.iloc[order].reset_index(drop=True)
        dupl = edb["gene_name"].duplicated(keep="first").to_numpy()
        edb = edb[~dupl].reset_index(drop=True)
        mtx = mtx[~dupl]
    else:
        # preProcessing.R:39-42 — dedup gene_name keeping FIRST, then reorder
        # edb to mtx's row order. mtx is NOT reordered/filtered.
        dupl = edb["gene_name"].duplicated(keep="first").to_numpy()
        edb = edb[~dupl].reset_index(drop=True)
        order = _order_match(edb[use_geneID].to_numpy(), mtx.index.to_numpy())
        edb = edb.iloc[order].reset_index(drop=True)

    # preProcessing.R:44-48 — cbind(edb, mtx). Row order = mtx order.
    mtx_reset = mtx.reset_index(drop=True)
    mtx_annot = pd.concat([edb, mtx_reset], axis=1)
    return mtx_annot


def _order_match(x, table):
    """R `order(match(x, table))`: the permutation of x's rows that sorts them
    by the position of each x-value within `table` (the mtx row order).

    match(x, table) gives, per x element, its first position in table; order()
    returns the indices that would sort those positions ascending. R's order()
    uses a stable sort for ties (radix on integers), which we mirror with a
    stable argsort. Assumes every x is present in table.
    """
    pos = {}
    for i, v in enumerate(table):
        if v not in pos:  # R match() returns the FIRST position (keep first)
            pos[v] = i
    match = [pos[v] for v in x]
    # stable argsort of match positions
    return sorted(range(len(match)), key=lambda i: match[i])
