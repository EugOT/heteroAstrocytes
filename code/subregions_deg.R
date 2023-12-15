#!/usr/bin/env R
# coding: utf-8

library(here)
library(RColorBrewer)
library(viridis)
library(tidyverse)
library(Seurat)
library(SeuratDisk)
library(scCustomize)

subregion_loom <- Connect(filename = snakemake@input[["loom"]], mode = "r")

subregion_srt <- as.Seurat(
    subregion_loom,
    features = "var_names",
    cells = "obs_names"
)
subregion_loom$close_all()
DefaultAssay(subregion_srt) <- "RNA"

subregion_srt <-
    Store_Palette_Seurat(
        seurat_object = subregion_srt,
        palette = rev(brewer.pal(n = 21, name = "Spectral")),
        palette_name = "expr_Colour_Pal"
    )

pacmap <- read.delim(snakemake@input[["pacmap_tsv"]]) %>% as.matrix()
rownames(pacmap) <- pacmap[, 1]
pacmap <- pacmap[, 2:3]
colnames(pacmap) <- paste0("PaCMAP_", 1:2)
subregion_srt[["pacmap"]] <- CreateDimReducObject(embeddings = pacmap, key = "PaCMAP_", assay = DefaultAssay(subregion_srt))

umap <- read.delim(snakemake@input[["umap_tsv"]]) %>% as.matrix()
rownames(umap) <- umap[, 1]
umap <- umap[, 2:3]
colnames(umap) <- paste0("UMAP_", 1:2)
subregion_srt[["umap"]] <- CreateDimReducObject(embeddings = umap, key = "UMAP_", assay = DefaultAssay(subregion_srt))

subregion_srt$clusters %>% table()
subregion_srt$clusters <- factor(subregion_srt$clusters)
Idents(subregion_srt) <- "clusters"


subregion_srt <- NormalizeData(subregion_srt)
subregion_srt <-
    FindVariableFeatures(
        subregion_srt,
        selection.method = "vst",
        nfeatures = 5000
    )

all_genes <- rownames(subregion_srt)
var_regex <- "^Hla-|^Ig[hjkl]|^Rna|^mt-|^Rp[sl]|^Hb[^(p)]|^Gm"
agg_genes <-
    GetAssayData(subregion_srt, slot = "counts", assay = "RNA") |> rowSums()
all_genes <-
    all_genes[agg_genes > 0.001 * ncol(subregion_srt)]
all_genes <-
    all_genes[str_detect(pattern = var_regex, string = all_genes, negate = TRUE)]

subregion_srt <- ScaleData(subregion_srt,
    features = all_genes,
    vars.to.regress = c(
        "log10GenesPerUMI"
    )
)

subregion_srt


markers_logreg <-
    FindAllMarkers(
        subregion_srt,
        assay = "RNA",
        verbose = FALSE,
        random.seed = reseed,
        only.pos = TRUE,
        min.pct = 0.1,
        base = 10,
        logfc.threshold = 0.1,
        densify = TRUE,
        test.use = "LR"
    )
write_csv(
    markers_logreg,
    here(
        snakemake@output[["deg_table_logreg"]]
    )
)
top10_markers <-
    Extract_Top_Markers(
        marker_dataframe = markers_logreg,
        num_genes = 10,
        named_vector = FALSE,
        make_unique = TRUE,
        rank_by = "avg_log10FC"
    )
top10_markers
pl_clst_dotplot <-
    try(
        {
            Clustered_DotPlot(
                seurat_object = subregion_srt,
                features = top10_markers,
                k = length(levels(subregion_srt)),
                ggplot_default_colors = TRUE,
                color_seed = reseed,
                seed = reseed
            )
        },
        silent = TRUE
    )

pdf(
    file = snakemake@output[["deg_top10_figures_logreg_dotplot"]],
    width = 16, height = 16
)
if (class(pl_clst_dotplot) != "try-error") pl_clst_dotplot[[2]]
dev.off()

# # markers
# try(
#     {
#         Iterate_FeaturePlot_scCustom(
#             seurat_object = subregion_srt,
#             gene_list = top10_markers,
#             single_pdf = TRUE,
#             reduction = "umap",
#             colors_use = viridis(
#                 n = 30,
#                 alpha = .55,
#                 direction = -1,
#                 option = "E"
#             ),
#             pt.size = 3,
#             alpha_exp = 0.45,
#             alpha_na_exp = 0.1,
#             file_name = snakemake@output[["deg_top10_figures_logreg"]]
#         )
#     },
#     silent = TRUE
# )

# if (!file.exists(snakemake@output[["deg_top10_figures_logreg"]])) {
#     top3_markers <-
#         Extract_Top_Markers(
#             marker_dataframe = markers_logreg,
#             num_genes = 3,
#             named_vector = FALSE,
#             make_unique = TRUE,
#             rank_by = "avg_log10FC"
#         )

#     try(
#         {
#             Iterate_FeaturePlot_scCustom(
#                 seurat_object = subregion_srt,
#                 gene_list = top3_markers,
#                 single_pdf = TRUE,
#                 reduction = "umap",
#                 colors_use = viridis(
#                     n = 30,
#                     alpha = .55,
#                     direction = -1,
#                     option = "E"
#                 ),
#                 pt.size = 3,
#                 alpha_exp = 0.45,
#                 alpha_na_exp = 0.1,
#                 file_name = snakemake@output[["deg_top10_figures_logreg"]]
#             )
#         },
#         silent = TRUE
#     )
# }

# if (!file.exists(snakemake@output[["deg_top10_figures_logreg"]])) {
#     pdf(
#         file = snakemake@output[["deg_top10_figures_logreg"]],
#         width = 12, height = 7.416
#     )
#     map(top10_markers, ~ FeaturePlot(subregion_srt, .x, pt.size = 2, order = T) + ggtitle(.x) + theme(plot.title = element_text(size = 24))) & scale_colour_gradientn(colours = subregion_srt@misc$expr_Colour_Pal)
#     dev.off()
# }


markers_MAST <-
    FindAllMarkers(
        subregion_srt,
        assay = "RNA",
        verbose = FALSE,
        random.seed = reseed,
        only.pos = TRUE,
        min.pct = 0.1,
        base = 10,
        logfc.threshold = 0.1,
        test.use = "MAST"
    )
write_csv(
    markers_MAST,
    here(
        snakemake@output[["deg_table_mast"]]
    )
)

top10_markers <-
    Extract_Top_Markers(
        marker_dataframe = markers_MAST,
        num_genes = 10,
        named_vector = FALSE,
        make_unique = TRUE,
        rank_by = "avg_log10FC"
    )

pl_clst_dotplot <-
    try(
        {
            Clustered_DotPlot(
                seurat_object = subregion_srt,
                features = top10_markers,
                k = length(levels(subregion_srt)),
                ggplot_default_colors = TRUE,
                color_seed = reseed,
                seed = reseed
            )
        },
        silent = TRUE
    )
pdf(
    file = snakemake@output[["deg_top10_figures_mast_dotplot"]],
    width = 16, height = 16
)
if (class(pl_clst_dotplot) != "try-error") pl_clst_dotplot[[2]]
dev.off()

# # markers
# try(
#     {
#         Iterate_FeaturePlot_scCustom(
#             seurat_object = subregion_srt,
#             gene_list = top10_markers,
#             single_pdf = TRUE,
#             reduction = "umap",
#             colors_use = viridis(
#                 n = 30,
#                 alpha = .55,
#                 direction = -1,
#                 option = "E"
#             ),
#             pt.size = 3,
#             alpha_exp = 0.45,
#             alpha_na_exp = 0.1,
#             file_name = snakemake@output[["deg_top10_figures_mast"]]
#         )
#     },
#     silent = TRUE
# )

# if (!file.exists(snakemake@output[["deg_top10_figures_mast"]])) {
#     top3_markers <-
#         Extract_Top_Markers(
#             marker_dataframe = markers_MAST,
#             num_genes = 3,
#             named_vector = FALSE,
#             make_unique = TRUE,
#             rank_by = "avg_log10FC"
#         )

#     try(
#         {
#             Iterate_FeaturePlot_scCustom(
#                 seurat_object = subregion_srt,
#                 gene_list = top3_markers,
#                 single_pdf = TRUE,
#                 reduction = "umap",
#                 colors_use = viridis(
#                     n = 30,
#                     alpha = .55,
#                     direction = -1,
#                     option = "E"
#                 ),
#                 pt.size = 3,
#                 alpha_exp = 0.45,
#                 alpha_na_exp = 0.1,
#                 file_name = snakemake@output[["deg_top10_figures_mast"]]
#             )
#         },
#         silent = TRUE
#     )
# }

# if (!file.exists(snakemake@output[["deg_top10_figures_mast"]])) {
#     pdf(
#         file = snakemake@output[["deg_top10_figures_mast"]],
#         width = 12, height = 7.416
#     )
#     map(top10_markers, ~ FeaturePlot(subregion_srt, .x, pt.size = 2, order = T) + ggtitle(.x) + theme(plot.title = element_text(size = 24))) & scale_colour_gradientn(colours = subregion_srt@misc$expr_Colour_Pal)
#     dev.off()
# }
