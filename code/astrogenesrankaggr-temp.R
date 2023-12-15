library(tidyverse)
library(Seurat)
library(RobustRankAggreg)

gn <- 45163
glist <- map(srt_list, VariableFeatures)
tc <- map_dbl(glist, ~length(.x)/gn)
granks <-
  aggregateRanks(
    glist = glist,
    N=gn,
    topCutoff = as.vector(tc))

filter(granks, Score < 5e-2) |> glimpse()
filter(granks, Score < 1e-3) |> glimpse()

