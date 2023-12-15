#!/usr/bin/env R
# coding: utf-8

library(tidyverse)
library(RobustRankAggreg)
# total number of distinct translatable genes in GENCODE Release M32
gn <- 45163
glist <- map(
  snakemake@input[["shared_anchor_genes"]],
  ~ read_tsv(.x) %>%
    slice_min(Score, n = 1000) %>%
    .$Name
)
tc <- map_dbl(glist, ~ length(.x) / gn)
granks <-
  aggregateRanks(
    glist = glist,
    N = gn,
    topCutoff = as.vector(tc)
  )
filter(granks, Score < 5e-2) |> glimpse()
filter(granks, Score < 1e-3) |> glimpse()

write_tsv(granks, file = snakemake@output[["aggregated_shared_genes"]])
