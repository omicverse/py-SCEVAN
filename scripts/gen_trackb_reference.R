# gen_trackb_reference.R — Track-B upstream references from frozen MGH106 subset.
# Run on <r-host> in ~/scevan_ref with SCEVAN installed in ~/Rlib.
# STAGE 1: frozen subset + annotate (T5) + preprocess (T6) references.
.libPaths(c("~/Rlib", .libPaths()))
suppressMessages({library(SCEVAN); library(dplyr)})
options(timeout = 900)

dir.create("ref/mgh106", recursive = TRUE, showWarnings = FALSE)

# --- MGH106 count matrix (GSE131928 glioblastoma, vignette source). ---
# Server cannot reach Dropbox (SSL); the .RData was downloaded locally and
# scp'd to ~/scevan_ref/MGH106_data.RData.
load("MGH106_data.RData")  # -> count_mtx
cat("MGH106 full:", nrow(count_mtx), "genes x", ncol(count_mtx), "cells\n")

# --- frozen deterministic subset (committed fixture input) ---
set.seed(2026)
ncell <- min(200, ncol(count_mtx))
sel <- sort(sample(ncol(count_mtx), ncell))
cm <- as.matrix(count_mtx[, sel])
cm <- cm[rowSums(cm) > 0, ]                 # drop all-zero genes (lean)
cat("subset:", nrow(cm), "genes x", ncol(cm), "cells\n")
write.table(cm, "ref/mgh106/mgh106_subset.tsv", sep = "\t", quote = FALSE, col.names = NA)

# --- T5: annotateGenes reference (no coordinate sort; row order = input) ---
annot <- SCEVAN:::annotateGenes(cm, organism = "human")
write.table(annot, "ref/mgh106/out_annot.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
cat("annotate:", nrow(annot), "genes,", ncol(annot) - 5, "cells\n")

# --- T6: preprocessingMtx reference (findConfident=FALSE -> yaGST-independent) ---
pp <- SCEVAN:::preprocessingMtx(cm, sample = "MGH106s", findConfident = FALSE, par_cores = 1)
norm_mtx <- pp[["count_mtx_norm"]]
annot2 <- pp[["count_mtx_annot"]]
write.table(norm_mtx, "ref/mgh106/preproc_norm.tsv", sep = "\t", quote = FALSE, col.names = NA)
write.table(annot2, "ref/mgh106/preproc_annot.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
cat("preproc_norm:", nrow(norm_mtx), "genes x", ncol(norm_mtx), "cells\n")

writeLines(capture.output(sessionInfo()), "ref/mgh106/sessionInfo.txt")
system("cd ref/mgh106 && md5sum mgh106_subset.tsv > input.md5 && gzip -f *.tsv")
cat("MGH106_STAGE1_DONE\n")
