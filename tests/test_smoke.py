def test_imports():
    import pyscevan
    assert set(["SCEVANConfig", "SCEVANResult", "pipeline_cna"]).issubset(pyscevan.__all__)
