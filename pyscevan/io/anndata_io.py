"""AnnData <-> SCEVAN I/O adapters (Task 11).

SCEVAN's ``pipelineCNA`` consumes a **genes x cells** count matrix (gene
symbols on the index, cell barcodes on the columns) plus an explicit list of
confident-normal cell names. AnnData stores the transpose (cells x genes) with
optional sparse ``X``. These helpers bridge the two conventions; pure Python,
no rpy2.
"""

from __future__ import annotations

import pandas as pd
from anndata import AnnData


def adata_to_count_mtx(adata: AnnData) -> pd.DataFrame:
    """Transpose an AnnData into a SCEVAN genes x cells count matrix.

    Parameters
    ----------
    adata
        Cells x genes AnnData. ``X`` may be dense or sparse; sparse is
        densified. Gene symbols are taken from ``adata.var_names``, cell names
        from ``adata.obs_names``.

    Returns
    -------
    DataFrame, genes (``var_names``) on the index, cells (``obs_names``) on the
    columns.
    """
    x = adata.X
    # Densify sparse matrices (scipy.sparse exposes .toarray()).
    if hasattr(x, "toarray"):
        x = x.toarray()
    # AnnData is cells x genes -> transpose to genes x cells.
    return pd.DataFrame(
        x.T,
        index=adata.var_names.astype(str),
        columns=adata.obs_names.astype(str),
    )


def normal_cells_from_obs(
    adata: AnnData,
    normal_key: str,
    normal_value: object | None = None,
) -> list[str]:
    """Extract confident-normal cell names from an ``obs`` column.

    Parameters
    ----------
    adata
        Cells x genes AnnData.
    normal_key
        Name of the ``obs`` column marking normal cells.
    normal_value
        If given, cells with ``obs[normal_key] == normal_value`` are normal.
        If ``None``, cells are normal when the value is truthy or equals the
        string ``"normal"`` (case-insensitive).

    Returns
    -------
    List of ``obs_names`` (cell barcodes) marked as normal, in ``obs`` order.
    """
    if normal_key not in adata.obs.columns:
        raise KeyError(
            f"normal_key {normal_key!r} not found in adata.obs "
            f"(columns: {list(adata.obs.columns)})"
        )
    col = adata.obs[normal_key]
    if normal_value is not None:
        mask = col == normal_value
    elif pd.api.types.is_bool_dtype(col) or pd.api.types.is_numeric_dtype(col):
        # Boolean / numeric flag column: truthy == normal.
        mask = col.astype(bool)
    else:
        # String / categorical column: the literal label "normal"
        # (case-insensitive). (A string column is "truthy" for every non-empty
        # value, so truthiness is meaningless here -- match the label instead.)
        mask = col.astype(str).str.lower() == "normal"
    return [str(c) for c in adata.obs_names[mask.to_numpy()]]
