# pyscevan vs R SCEVAN — benchmark summary

Patients: **18**  ·  py↔R consistency only (no external truth; the 3CA `Malignant` label is itself an algorithm output, not a reference).

## Four dimensions

| Dimension | Metric | Mean | Median | Min |
|---|---|---|---|---|
| 1. Speed | speedup (R/py) | 17.33× | 18.34× | 9.62× |
| 2. CNA matrix | Spearman ρ (overall) | 1.0000 | 1.0000 | 1.0000 |
| 2. CNA matrix | Spearman ρ (per-cell median) | 1.0000 | 1.0000 | 1.0000 |
| 4. Tumour calls | ARI (py vs R) | 1.0000 | 1.0000 | 1.0000 |
| 4. Tumour calls | Jaccard (py vs R) | 1.0000 | 1.0000 | 1.0000 |

**3. Functional coverage** (static, see `NAMESPACE_PARITY.md`): 4/11 R exports ported (+5 internal helpers); 7/11 not ported (yaGST auto-normal = Phase 1.5; multi-sample / subclone / plotting / oncoprint = Phase 2).

## Per-patient

| Patient | speedup | CNA ρ | ρ cell-median | tumour ARI | Jaccard | overlap | py-only | r-only | py s | r s |
|---|---|---|---|---|---|---|---|---|---|---|
| Gao2021_Breast__DCIS1 | 11.07× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1099 | 0 | 0 | 75.9 | 840.5 |
| Gao2021_Breast__TNBC1 | 9.62× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 797 | 0 | 0 | 64.0 | 615.5 |
| Gao2021_Breast__TNBC2 | 9.94× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 618 | 0 | 0 | 49.9 | 496.4 |
| Gao2021_Breast__TNBC3 | 14.04× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 180 | 0 | 0 | 16.8 | 235.9 |
| Kim2020_Lung__P0019 | 17.40× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1850 | 0 | 0 | 274.2 | 4772.2 |
| Kim2020_Lung__P0034 | 18.74× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2257 | 0 | 0 | 164.4 | 3080.0 |
| Kim2020_Lung__P1028 | 19.88× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2475 | 0 | 0 | 459.2 | 9131.4 |
| Lee2020_Colorectal__SMC09 | 15.69× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1795 | 0 | 0 | 136.8 | 2146.1 |
| Lee2020_Colorectal__SMC16 | 24.16× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1718 | 0 | 0 | 148.9 | 3596.4 |
| Lee2020_Colorectal__SMC21 | 18.70× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1661 | 0 | 0 | 109.1 | 2039.5 |
| MGH106s | n/a× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 136 | 0 | 0 | 5.0 | n/a |
| Obradovic2021_Kidney__Patient2 | 21.08× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 710 | 0 | 0 | 155.0 | 3266.6 |
| Obradovic2021_Kidney__Patient4 | 19.57× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2699 | 0 | 0 | 599.3 | 11726.5 |
| Obradovic2021_Kidney__Patient5 | 18.78× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2183 | 0 | 0 | 336.1 | 6312.1 |
| Qian2020_Ovarian__11 | 24.74× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 803 | 0 | 0 | 1221.3 | 30212.6 |
| Qian2020_Ovarian__12 | 16.06× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1221 | 0 | 0 | 292.8 | 4703.9 |
| Qian2020_Ovarian__13 | 16.77× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1324 | 0 | 0 | 241.8 | 4055.3 |
| Qian2020_Ovarian__14 | 18.34× | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3378 | 0 | 0 | 406.0 | 7447.5 |
