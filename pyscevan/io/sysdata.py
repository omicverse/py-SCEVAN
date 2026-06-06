"""Loaders for bundled SCEVAN sysdata resources (annotation, gene sets, chrom sizes)."""

from importlib.resources import files

import pandas as pd

_D = files("pyscevan.data")


def load_annotation(organism: str = "human") -> pd.DataFrame:
    fn = "EnsDB_Hsapiens_v86.tsv" if organism == "human" else "EnsDb_Mmusculus_v79.tsv"
    df = pd.read_csv(_D / fn, sep="\t")
    return df  # cols: seqnames,start,end,width,strand,gene_id,gene_name,gene_biotype,...


def load_gene_sets(organism: str = "human") -> dict[str, list[str]]:
    fn = "geneSet.gmt" if organism == "human" else "geneSetMouse.gmt"
    out: dict[str, list[str]] = {}
    for line in (_D / fn).read_text().splitlines():
        p = line.split("\t")
        out[p[0]] = p[2:]
    return out


def load_chrom_sizes(organism: str = "human") -> pd.DataFrame:
    fn = "sizeGRCh38.tsv" if organism == "human" else "sizeGRCm39.tsv"
    return pd.read_csv(_D / fn, sep="\t")
