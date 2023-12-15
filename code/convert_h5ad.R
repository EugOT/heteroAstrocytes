#!/usr/bin/env R
# coding: utf-8

library(reticulate)
library(anndata)
library(sceasy)

library(Seurat)
library(SeuratDisk)

sc <- import("scanpy", convert = FALSE)

srt <- LoadH5Seurat(
  file = here::here(
    snakemake@input[["h5ad"]]
  )
)

DefaultAssay(srt) <- "RNA"

adata <- convertFormat(srt, from = "seurat", to = "anndata", main_layer = "counts", drop_single_values = FALSE)
print(adata)
adata$write(snakemake@output[["h5ad"]])
