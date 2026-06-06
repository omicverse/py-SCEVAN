# gen_r_reference_ext.R — extended vegaMC fixtures to strengthen the white-box oracle
# (codex bootstrap review §1: 3 base cases let buggy kernels pass).
# Run on <r-host> in ~/scevan_ref:  Rscript gen_r_reference_ext.R
setwd("~/scevan_ref"); dyn.load("src/vegaMC.so")
source("vegaMC.R"); source("qvalue.R")
dir.create("ref/vegamc", recursive=TRUE, showWarnings=FALSE)
dir.create("out", showWarnings=FALSE)

emit <- function(id, Name, Chr, Position, V){
  mtx <- data.frame(Name=Name, Chr=Chr, Position=Position, V, check.names=FALSE)
  write.table(mtx, sprintf("ref/vegamc/in_%s.tsv", id), sep="\t", quote=FALSE, row.names=FALSE)
  tryCatch({
    seg <- vegaMC_R(mtx, output_file_name=sprintf("out/%s_vega", id), beta=0.5, html=FALSE, getGenes=FALSE)
    write.table(seg, sprintf("ref/vegamc/seg_%s.tsv", id), sep="\t", quote=FALSE, row.names=FALSE)
    cat(sprintf("case %s: %d seg, chrs={%s}, sizes={%s}\n", id, nrow(seg),
                paste(unique(seg$Chr), collapse=","), paste(seg$`Probe Size`, collapse=",")))
  }, error=function(e) cat(sprintf("case %s: ERROR %s\n", id, conditionMessage(e))))
}

## c4 — multi-chromosome: per-chr segmentation, blocks at chr boundaries, no cross-chr merge
mk_chr <- function(chr, n, blocks, ns=5, seed){
  set.seed(seed); pos <- sort(sample(1:2e8, n)); m <- matrix(rnorm(n*ns,0,0.15), n, ns)
  for(b in blocks) m[b$rows,] <- m[b$rows,] + b$delta
  list(chr=rep(chr,n), pos=pos, m=m, name=paste0("c",chr,"_g",1:n))
}
a <- mk_chr(1, 100, list(list(rows=1:30,  delta=-0.8)), seed=11)   # loss at START of chr1
b <- mk_chr(2,  80, list(list(rows=60:80, delta= 0.9)), seed=12)   # gain at END of chr2
d <- mk_chr(3,  60, list(),                              seed=13)   # neutral chr3
emit("c4", c(a$name,b$name,d$name), c(a$chr,b$chr,d$chr), c(a$pos,b$pos,d$pos), rbind(a$m,b$m,d$m))

## c5 — exact ties: identical rows within a level -> adjacent priority 0 -> locks smaller-id tie-break
ns <- 5; lvl <- function(v,k) matrix(v, k, ns)
V5 <- rbind(lvl(0.0,20), lvl(0.6,20), lvl(0.0,20)); n5 <- nrow(V5)
set.seed(105); emit("c5", paste0("g",1:n5), rep(1L,n5), sort(sample(1:2e8,n5)), V5)

## c6 — focal min_region_bp_size boundary: span==1000 dropped (strict >), span==1002 kept
V6 <- rbind(matrix(0.8,5,5), matrix(-0.8,5,5))
pos6 <- c(1000,1100,1200,1300,1999,  3000,3100,3200,3300,4001)     # spans 1000 / 1002
emit("c6", paste0("g",1:10), rep(1L,10), pos6, V6)

## c7 — all-neutral low-noise (stop condition w/ small std)
set.seed(107); emit("c7", paste0("g",1:80), rep(1L,80), sort(sample(1:2e8,80)), matrix(rnorm(80*5,0,0.05),80,5))
## c7z — exactly zero variance (std==0 stop-threshold edge)
emit("c7z", paste0("g",1:40), rep(1L,40), sort(sample(1:2e8,40)), matrix(0.0,40,5))

## c8 — tiny chromosome: 2-probe chr1 alongside a normal chr2 (heap of 1 node; >=2 probes safe)
set.seed(108)
t1 <- matrix(rnorm(2*5,0,0.1),2,5)
t2 <- matrix(rnorm(40*5,0,0.15),40,5); t2[10:25,] <- t2[10:25,] + 0.7
emit("c8", c("c1_g1","c1_g2",paste0("c2_g",1:40)), c(1L,1L,rep(2L,40)),
     c(sort(sample(1:1e6,2)), sort(sample(1:2e8,40))), rbind(t1,t2))

## c9 — partial-flat: flat chr1 (std=0, trivial segs dropped) + valid chr2 (loss block).
## Proves a single flat chromosome does NOT abort vegaMC (only the valid chr emits);
## locks the "raise only on whole-output-empty" guard (codex T3 review CONCERN 5).
set.seed(109)
c1v <- matrix(0.0, 30, 5)
c2v <- matrix(rnorm(60*5,0,0.15),60,5); c2v[20:45,] <- c2v[20:45,] - 0.8
emit("c9", c(paste0("a",1:30),paste0("b",1:60)), c(rep(1L,30),rep(2L,60)),
     c(sort(sample(1:5e7,30)), sort(sample(1:2e8,60))), rbind(c1v,c2v))

cat("DONE_EXT\n")
