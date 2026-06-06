"""``pyscevan`` CLI entry point (typer-based).

Installed as the ``pyscevan`` console script via ``pyproject.toml``::

    [project.scripts]
    pyscevan = "pyscevan.cli:app"

The single MVP command ``run-h5ad`` drives the user-normal ``pipeline_cna``
path off an AnnData (.h5ad): it pulls the genes x cells count matrix and the
confident-normal cell list out of the AnnData and writes the three result
tables to ``--output``.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="pyscevan — Python rewrite of SCEVAN (MVP)")


@app.command()
def version() -> None:
    """Print the installed pyscevan version."""
    from importlib.metadata import PackageNotFoundError, version as _v

    try:
        typer.echo(_v("pyscevan"))
    except PackageNotFoundError:
        typer.echo("0.0.0+unknown")


@app.command("run-h5ad")
def run_h5ad(
    input: Path = typer.Option(  # noqa: A002 (mirror pycopykat CLI flag name)
        ..., "--input", help="input .h5ad file (cells x genes)"
    ),
    normal_key: str = typer.Option(
        ..., "--normal-key", help="obs column marking confident-normal cells"
    ),
    output: Path = typer.Option(..., "--output", help="output directory"),
    normal_value: str | None = typer.Option(
        None,
        "--normal-value",
        help="obs[normal_key] value marking normal (default: bool/numeric "
        "column = truthy; string column = literal 'normal' label only)",
    ),
    sample: str = typer.Option("sample", "--sample", help="sample label"),
    beta_vega: float = typer.Option(
        0.5, "--beta-vega", help="vegaMC segmentation beta for classification"
    ),
) -> None:
    """Run the MVP SCEVAN CNA pipeline on one AnnData and write result tables."""
    import anndata as ad

    from pyscevan import pipeline_cna
    from pyscevan.io.anndata_io import adata_to_count_mtx, normal_cells_from_obs

    adata = ad.read_h5ad(input)
    count_mtx = adata_to_count_mtx(adata)
    normals = normal_cells_from_obs(adata, normal_key, normal_value=normal_value)

    res = pipeline_cna(
        count_mtx,
        norm_cell=normals,
        sample=sample,
        beta_vega=beta_vega,
        subclones=False,
        clonal_cn=True,
    )

    output.mkdir(parents=True, exist_ok=True)
    res.class_df.to_csv(output / f"{sample}_classDf.tsv", sep="\t")
    if res.clonal_cn is not None:
        res.clonal_cn.to_csv(
            output / f"{sample}_clonalCN.tsv", sep="\t", index=False
        )
    if res.cna_matrix is not None:
        res.cna_matrix.to_csv(
            output / f"{sample}_CNAmat.tsv.gz", sep="\t", index=False
        )

    n_tumor = int((res.class_df["class"] == "tumor").sum())
    n_normal = int((res.class_df["class"] == "normal").sum())
    typer.echo(
        f"{sample}: {n_tumor} tumor / {n_normal} normal cells "
        f"(of {len(res.class_df)} classified); wrote tables under {output}"
    )


if __name__ == "__main__":
    app()
