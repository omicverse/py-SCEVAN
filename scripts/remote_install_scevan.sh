#!/usr/bin/env bash
# remote_install_scevan.sh — Track B: full-stack SCEVAN install on <r-host> (Task R1).
# Long-running -> launch inside remote tmux. Installs to a personal lib (~/Rlib).
set -euo pipefail
mkdir -p ~/Rlib
export R_LIBS_USER=~/Rlib
# Drop stale Windows-built objects so install_local recompiles the C kernel cleanly.
rm -f ~/scevan_ref/SCEVAN-R/src/*.o ~/scevan_ref/SCEVAN-R/src/*.so
Rscript -e '
.libPaths("~/Rlib")
options(repos = c(CRAN="https://cloud.r-project.org"))
if(!requireNamespace("BiocManager",quietly=TRUE)) install.packages("BiocManager")
if(!requireNamespace("remotes",quietly=TRUE)) install.packages("remotes")
bioc <- c("scran","fgsea","EnsDb.Hsapiens.v86","ggtree")
cran <- c("ape","tidytree","Rtsne","pheatmap","parallelDist","igraph","cluster","clue",
          "dplyr","forcats","ggrepel","RColorBrewer")
BiocManager::install(bioc, ask=FALSE, update=FALSE)
need <- setdiff(cran, rownames(installed.packages()))
if(length(need)) install.packages(need)
remotes::install_github("miccec/yaGST", upgrade="never")
# Install SCEVAN from the scp-ed local source (faithful upstream version), compiling the C kernel.
remotes::install_local("~/scevan_ref/SCEVAN-R", upgrade="never", force=TRUE)
cat("SCEVAN installed:", requireNamespace("SCEVAN",quietly=TRUE), "\n")
'
echo "R1_DONE"
