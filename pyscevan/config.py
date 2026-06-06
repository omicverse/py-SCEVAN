from dataclasses import dataclass


@dataclass
class SCEVANConfig:  # noqa: N801 — mirror R opts
    sample: str = ""
    beta_vega: float = 0.5
    organism: str = "human"
    ngenes_chr: int = 5          # noqa: N803-style R fidelity
    perc_genes: float = 10.0
    SUBCLONES: bool = False      # MVP 默认 False（Phase 2 才启用）；若设 True，pipeline 运行时显式 warn 被忽略
    ClonalCN: bool = True
    FIXED_NORMAL_CELLS: bool = False
    SCEVANsignatures: bool = True
    output_dir: str = "./output"
