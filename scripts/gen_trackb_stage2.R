# gen_trackb_stage2.R — Track-B Stage 2: classify (T9) + compute_cna (T8) +
# EM CN-call (T7) + pipeline classDf (T10) references from the frozen MGH106
# subset. Run on <r-host> in ~/scevan_ref with SCEVAN in ~/Rlib.
# MVP path: user-provided normal -> we FREEZE the auto-found normals once, then
# all downstream refs are deterministic. No plotting / no subclones.
.libPaths(c("~/Rlib", .libPaths()))
suppressMessages({library(SCEVAN); library(dplyr)})
setwd("~/scevan_ref")
RD <- "ref/mgh106"
OUT <- file.path(RD, "scevan_out"); dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

cm <- as.matrix(read.table(gzfile(file.path(RD, "mgh106_subset.tsv.gz")),
                           header = TRUE, row.names = 1, sep = "\t", check.names = FALSE))
cat("cm:", nrow(cm), "x", ncol(cm), "\n")

## --- preprocess + freeze auto-found normal cells (deterministic via set.seed) ---
res_proc <- SCEVAN:::preprocessingMtx(cm, sample = "MGH106s", par_cores = 1, findConfident = TRUE,
                                      output_dir = OUT)
norm_cell <- names(res_proc$norm_cell)
cat("auto-found normal cells:", length(norm_cell), "\n")
writeLines(norm_cell, file.path(RD, "norm_cell.txt"))

## --- T9: classifyTumorCells with the frozen normal set (user-normal path) ---
res_class <- SCEVAN:::classifyTumorCells(
  res_proc$count_mtx_norm, res_proc$count_mtx_annot, sample = "MGH106s", par_cores = 1,
  norm_cell_names = norm_cell, SEGMENTATION_CLASS = TRUE, SMOOTH = TRUE,
  beta_vega = 0.5, FIXED_NORMAL_CELLS = FALSE, output_dir = OUT)
cat("tum_cells:", length(res_class$tum_cells), "\n")
write.table(res_class$CNAmat, gzfile(file.path(RD, "classify_CNAmat.tsv.gz")),
            sep = "\t", quote = FALSE, col.names = NA)
writeLines(res_class$tum_cells, file.path(RD, "classify_tum_cells.txt"))

## --- T8: computeCNAmtx standalone (small deterministic input) ---
set.seed(7)
small <- matrix(rnorm(120 * 8, 0, 0.3), 120, 8); small[30:70, ] <- small[30:70, ] + 0.6
rownames(small) <- paste0("g", 1:120); colnames(small) <- paste0("c", 1:8)
breaks8 <- c(1, 30, 71, 100, 120)          # break starts (1-based), incl 1 and n
segmAlt8 <- c(TRUE, TRUE, FALSE, TRUE)      # length == #segments == length(breaks)-1
cna8 <- SCEVAN:::computeCNAmtx(small, breaks8, par_cores = 1, segmAlt8)
rownames(cna8) <- rownames(small); colnames(cna8) <- colnames(small)
write.table(small, gzfile(file.path(RD, "t8_in_mtx.tsv.gz")), sep = "\t", quote = FALSE, col.names = NA)
writeLines(as.character(breaks8), file.path(RD, "t8_breaks.txt"))
writeLines(as.character(as.integer(segmAlt8)), file.path(RD, "t8_segmAlt.txt"))
write.table(cna8, gzfile(file.path(RD, "t8_out_cna.tsv.gz")), sep = "\t", quote = FALSE, col.names = NA)
cat("T8 computeCNAmtx done:", nrow(cna8), "x", ncol(cna8), "\n")

## --- T7: getCNcall on tumor cells (mirrors getClonalCNProfile, CLONAL, beta=3) ---
tum_mtx <- res_class$CNAmat[, res_class$tum_cells]
mtx_vega <- cbind(res_class$CNAmat[, 1:3], tum_mtx)
colnames(mtx_vega)[1:3] <- c("Name", "Chr", "Position")
breaks_tumor <- SCEVAN:::getBreaksVegaMC(mtx_vega, res_class$CNAmat[, 3], "MGH106sClonal",
                                         beta_vega = 3, output_dir = OUT)
CNV <- SCEVAN:::getCNcall(tum_mtx, res_proc$count_mtx_annot, breaks_tumor,
                         sample = "MGH106s", CLONAL = TRUE, par_cores = 1)
write.table(CNV, file.path(RD, "t7_clonalCN.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(as.character(breaks_tumor), file.path(RD, "t7_breaks_tumor.txt"))
write.table(tum_mtx, gzfile(file.path(RD, "t7_in_tum_mtx.tsv.gz")), sep = "\t", quote = FALSE, col.names = NA)
write.table(res_proc$count_mtx_annot, gzfile(file.path(RD, "t7_annot.tsv.gz")),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("T7 getCNcall done; breaks_tumor:", length(breaks_tumor), "; CNV rows:", nrow(CNV), "\n")

## --- T10: classDf (filtered/normal/tumor) per pipelineCNA.R:52-54 ---
classDf <- data.frame(class = rep("filtered", ncol(res_proc$count_mtx)),
                      row.names = colnames(res_proc$count_mtx))
classDf[colnames(res_class$CNAmat)[-(1:3)], "class"] <- "normal"
classDf[res_class$tum_cells, "class"] <- "tumor"
classDf[res_class$confidentNormal, "confidentNormal"] <- "yes"   # pipelineCNA.R:55
write.table(classDf, file.path(RD, "t10_classDf.tsv"), sep = "\t", quote = FALSE, col.names = NA)
cat("T10 classDf:", paste(names(table(classDf$class)), table(classDf$class), collapse = " "), "\n")

writeLines(capture.output(sessionInfo()), file.path(RD, "sessionInfo_stage2.txt"))
cat("STAGE2_DONE\n")
