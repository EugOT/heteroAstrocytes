#!/usr/bin/env R
# coding: utf-8

library(tidyverse)
library(RobustRankAggreg)
# total number of distinct translatable genes in GENCODE Release M32
gn <- 45163

slice_genes <- function(path) {
  dat <- read_tsv(path)
  genes <- dat %>%
    filter(highly_variable_intersection) %>%
    slice_min(highly_variable_rank, n = 2000) %>%
    .$`...1`
  return(genes)
}

glist <- map(
  snakemake@input[["anchor_genes"]],
  ~ slice_genes(.x)
)
glist
tc <- map_dbl(glist, ~ length(.x) / gn)
granks <-
  aggregateRanks(
    glist = glist,
    N = gn,
    topCutoff = as.vector(tc)
  )
filter(granks, Score < 5e-2) |> glimpse()
filter(granks, Score < 1e-3) |> glimpse()

write_tsv(granks, file = snakemake@output[["shared_anchor_genes"]])
