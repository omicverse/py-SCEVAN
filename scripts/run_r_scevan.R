# run_r_scevan.R -- R SCEVAN reference for one patient (benchmark route A).
#
# Run on <r-host> with SCEVAN in ~/Rlib:
#   Rscript run_r_scevan.R <counts.tsv[.gz]> <out_dir> <sam_name> [organism]
#
# Route A: R's own findConfident=TRUE auto-detects the confident-normal set; we
# FREEZE it to norm_cell.txt and hand the identical list to pyscevan, so the two
# pipelines are compared on the same normals (auto-detect is unported = Phase
# 1.5). The TIMED region mirrors pyscevan.pipeline_cna exactly:
#   preprocessingMtx(findConfident=FALSE) + classifyTumorCells + clonal getCNcall
# The auto-normal detection (untimed setup) is excluded from the speedup, since
# pyscevan does not perform it.
#
# Outputs (uniform with run_py_scevan.py) into <out_dir>:
#   <sam>_r_CNAmat.tsv.gz   gene_id/seqnames/end + cells (relativised body)
#   <sam>_r_tum_cells.txt   one tumour cell per line
#   <sam>_r_classDf.tsv     index=cell, columns class/confidentNormal
#   <sam>_r_runinfo.txt     elapsed_sec=<timed region wall clock>
#   norm_cell.txt           R auto-detected normals (fed to py)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: run_r_scevan.R <counts> <out_dir> <sam> [organism]")
counts_path <- args[1]; out_dir <- args[2]; sam <- args[3]
organism <- if (length(args) >= 4) args[4] else "human"

.libPaths(c("~/Rlib", .libPaths()))
suppressMessages({library(SCEVAN); library(dplyr)})
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
scratch <- file.path(out_dir, "scevan_scratch"); dir.create(scratch, showWarnings = FALSE)

con <- if (grepl("\\.gz$", counts_path)) gzfile(counts_path) else counts_path
cm <- as.matrix(read.table(con, header = TRUE, row.names = 1, sep = "\t",
                           check.names = FALSE))
cat(sprintf("[R] %s: %d genes x %d cells\n", sam, nrow(cm), ncol(cm)))

## --- setup (UNTIMED): R auto-detects confident normals (findConfident=TRUE) ---
res_setup <- SCEVAN:::preprocessingMtx(cm, sample = sam, par_cores = 1,
                                       findConfident = TRUE, output_dir = scratch)
norm_cell <- names(res_setup$norm_cell)
cat(sprintf("[R] auto-found %d confident-normal cells\n", length(norm_cell)))
writeLines(norm_cell, file.path(out_dir, "norm_cell.txt"))

## --- TIMED region: mirrors pyscevan.pipeline_cna (user-normal path) ---
t0 <- proc.time()[["elapsed"]]
res_proc <- SCEVAN:::preprocessingMtx(cm, sample = sam, par_cores = 1,
                                      findConfident = FALSE, output_dir = scratch)
res_class <- SCEVAN:::classifyTumorCells(
  res_proc$count_mtx_norm, res_proc$count_mtx_annot, sample = sam, par_cores = 1,
  norm_cell_names = norm_cell, SEGMENTATION_CLASS = TRUE, SMOOTH = TRUE,
  beta_vega = 0.5, FIXED_NORMAL_CELLS = FALSE, output_dir = scratch)
tum_cells <- res_class$tum_cells
# clonal CN (getClonalCNProfile equivalent, beta_vega=3 on tumour cells)
tum_mtx <- res_class$CNAmat[, tum_cells, drop = FALSE]
mtx_vega <- cbind(res_class$CNAmat[, 1:3], tum_mtx)
colnames(mtx_vega)[1:3] <- c("Name", "Chr", "Position")
breaks_tumor <- SCEVAN:::getBreaksVegaMC(mtx_vega, res_class$CNAmat[, 3],
                                         paste0(sam, "Clonal"), beta_vega = 3,
                                         output_dir = scratch)
clonal_cn <- SCEVAN:::getCNcall(tum_mtx, res_proc$count_mtx_annot, breaks_tumor,
                                sample = sam, CLONAL = TRUE, par_cores = 1)
elapsed <- proc.time()[["elapsed"]] - t0
cat(sprintf("[R] timed region: %.2fs; %d tumour cells\n", elapsed, length(tum_cells)))

## --- write uniform artifacts (gene_id/seqnames/end + cells) ---
cna <- res_class$CNAmat
colnames(cna)[1:3] <- c("gene_id", "seqnames", "end")
gz <- gzfile(file.path(out_dir, sprintf("%s_r_CNAmat.tsv.gz", sam)))
write.table(cna, gz, sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(tum_cells, file.path(out_dir, sprintf("%s_r_tum_cells.txt", sam)))

# classDf (filtered/normal/tumor) per pipelineCNA.R:52-55
classDf <- data.frame(class = rep("filtered", ncol(res_proc$count_mtx_norm)),
                      row.names = colnames(res_proc$count_mtx_norm))
classDf[colnames(res_class$CNAmat)[-(1:3)], "class"] <- "normal"
classDf[tum_cells, "class"] <- "tumor"
classDf$confidentNormal <- NA
classDf[res_class$confidentNormal, "confidentNormal"] <- "yes"
write.table(classDf, file.path(out_dir, sprintf("%s_r_classDf.tsv", sam)),
            sep = "\t", quote = FALSE, col.names = NA)

writeLines(c(sprintf("elapsed_sec=%.4f", elapsed),
             sprintf("n_tumor=%d", length(tum_cells)),
             sprintf("n_cells=%d", ncol(res_class$CNAmat) - 3)),
           file.path(out_dir, sprintf("%s_r_runinfo.txt", sam)))
cat(sprintf("[R] %s DONE\n", sam))
