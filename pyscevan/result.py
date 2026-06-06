from dataclasses import dataclass

import pandas as pd


@dataclass
class SCEVANResult:
    class_df: pd.DataFrame            # cell -> class (filtered/normal/tumor)
    cna_matrix: pd.DataFrame | None = None
    clonal_cn: pd.DataFrame | None = None
