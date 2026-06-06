# gen_r_reference.R — generate vegaMC R-oracle reference cases (Task R0).
# Run on <r-host> in ~/scevan_ref:  Rscript gen_r_reference.R
# Uses the compiled vegaMC.so kernel only (no Bioconductor; getGenes/html FALSE).
setwd("~/scevan_ref"); dyn.load("src/vegaMC.so")
source("vegaMC.R"); source("qvalue.R")
dir.create("ref/vegamc", recursive=TRUE, showWarnings=FALSE)
dir.create("out", showWarnings=FALSE)        # C fopen needs output dir to exist (codex blocker 5)

gen_case <- function(id, n, ns, blocks, seed){
  set.seed(seed)
  pos <- sort(sample(1:2e8, n)); base <- matrix(rnorm(n*ns,0,0.15), n, ns)
  for(b in blocks) base[b$rows,] <- base[b$rows,] + b$delta
  mtx <- data.frame(Name=paste0("g",1:n), Chr=rep(1L,n), Position=pos, base, check.names=FALSE)
  write.table(mtx, sprintf("ref/vegamc/in_%s.tsv", id), sep="\t", quote=FALSE, row.names=FALSE)
  seg <- vegaMC_R(mtx, output_file_name=sprintf("out/%s_vega", id),
                  beta=0.5, html=FALSE, getGenes=FALSE)
  write.table(seg, sprintf("ref/vegamc/seg_%s.tsv", id), sep="\t", quote=FALSE, row.names=FALSE)
  cat(sprintf("case %s: %d segments\n", id, nrow(seg)))
}
gen_case("c1", 300, 5, list(list(rows=80:130,delta=-0.8), list(rows=200:250,delta=0.7)), 42)
gen_case("c2", 500, 8, list(list(rows=100:180,delta=-0.5), list(rows=300:330,delta=1.0)), 7)
gen_case("c3", 150, 3, list(list(rows=40:60,delta=0.4)), 99)

# reproducibility manifest (codex blocker 5)
writeLines(capture.output(sessionInfo()), "ref/vegamc/sessionInfo.txt")
system("gcc --version | head -1 > ref/vegamc/gcc.txt")
cat("DONE\n")
