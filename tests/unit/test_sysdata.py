from pyscevan.io import sysdata


def test_annotation_shape():
    df = sysdata.load_annotation("human")
    assert df.shape == (63970, 11)
    assert {"gene_name", "seqnames", "start", "end"}.issubset(df.columns)


def test_gene_sets():
    gs = sysdata.load_gene_sets("human")
    assert len(gs) == 12 and "Tcell" in gs and len(gs["Tcell"]) > 0


def test_chrom_sizes():
    sz = sysdata.load_chrom_sizes("human")
    assert sz.shape[0] == 22
