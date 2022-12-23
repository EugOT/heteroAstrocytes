{r transfer-campbell-2017-1-ageing}
campbell_2017.1.anchors.hypoth <-
  FindTransferAnchors(
    reference = hypoth.query.campbell_2017.1,
    query = combined_sct,
    normalization.method = "SCT",
    dims = 1:npcs,
    k.anchor = 10,
    k.score = 50,
    k.filter = 100,
    max.features = 500,
    n.trees = 100,
    reference.reduction = "pca")

campbell_2017.1.predictions.hypoth <-
  TransferData(
    anchorset = campbell_2017.1.anchors.hypoth,
    refdata = hypoth.query.campbell_2017.1$k_tree,
    dims = 1:npcs)

combined_sct <-
  MapQuery(
    anchorset = campbell_2017.1.anchors.hypoth,
    reference = hypoth.query.campbell_2017.1,
    query = combined_sct,
    refdata = list(arc_campbell_2017_1_mrt = "k_tree"),
    new.reduction.name = "campbell2017.1.umap",
    reference.reduction = "pca",
    reduction.model = "umap")



{r pl-transfer-campbell-2017-1-ageing, fig.asp=0.206, fig.width = 16}
p.hypoth.campbell_2017.1 <-
  DimPlot(
    hypoth.query.campbell_2017.1,
    reduction = "umap",
    group.by = "predicted.k_tree",
    label = TRUE,
    label.size = 3,
    repel = TRUE
  ) +
  NoLegend() +
  ggtitle("Campbell et al. 2017 batch-1 transferred labels")

p.orig.campbell_2017.1 <-
  DimPlot(
    hypoth.query.campbell_2017.1,
    reduction = "umap",
    group.by = "k_tree",
    label = TRUE,
    label.size = 3,
    repel = TRUE
  ) +
  NoLegend() +
  ggtitle("Campbell et al. 2017 batch-1 annotations")

p.campbell_2017.1.hypoth <-
  DimPlot(
    combined_sct,
    reduction = "campbell2017.1.umap",
    group.by = "predicted.arc_campbell_2017_1_mrt",
    label = TRUE,
    label.size = 3,
    repel = TRUE
  ) +
  NoLegend() +
  ggtitle("Reference backward-transferred labels from Campbell et al. 2017 batch-1")

p.hypoth.campbell_2017.1 +
  p.orig.campbell_2017.1 +
  p.campbell_2017.1.hypoth & scale_x_continuous(
    limits = c(
      floor(min(
        Embeddings(hypoth.query.campbell_2017.1,
                   "umap")[,"UMAP_1"])),
      ceiling(max(
        Embeddings(hypoth.query.campbell_2017.1,
                   "umap")[,"UMAP_1"]))
    )
  ) & scale_y_continuous(
    limits = c(
      floor(min(
        Embeddings(hypoth.query.campbell_2017.1,
                   "umap")[,"UMAP_2"])),
      ceiling(max(
        Embeddings(hypoth.query.campbell_2017.1,
                   "umap")[,"UMAP_2"]))
    )
  )
