DeriveKTree <- function(srt, n.pcs = n_pcs, vseed = reseed, n.cores = n_cores) {
  plan("multiprocess", workers = n.cores)

  srt <-
    SCTransform(
      srt,
      vst.flavor = "v2",
      variable.features.n = 8000,
      vars.to.regress = "percent_mt",
      return.only.var.genes = FALSE,
      seed.use = vseed,
      verbose = FALSE
    ) %>%
    RunPCA(
      npcs = n.pcs,
      seed.use = vseed,
      verbose = FALSE) %>%
    RunUMAP(
      reduction = "pca",
      return.model = TRUE,
      umap.method = "uwot-learn",
      densmap = TRUE,
      dens.lambda = 1,
      dens.frac = 0.25,
      dims = 1:n.pcs,
      n.epochs = 1000,
      n.neighbors = 15L,
      min.dist = 0.1,
      spread = 2,
      metric = "cosine",
      seed.use = vseed,
      verbose = FALSE) %>%
    FindNeighbors(
      object = .,
      features = VariableFeatures(.),
      k.param = 12,
      annoy.metric = "cosine",
      n.trees = 100,
      verbose = FALSE)

  resolutions <-
    modularity_event_sampling(
      A = srt@graphs$SCT_snn,
      n.res = 30,
      gamma.min = 0.8,
      gamma.max = 4.00001
    ) # sample based on the similarity matrix

  srt <- FindClusters(
    srt, algorithm = 4, method = "igraph",
    resolution = resolutions, random.seed = vseed,
    verbose = FALSE)

  out <-  mrtree(
    srt,
    prefix = 'SCT_snn_res.',
    n.cores = n.cores,
    consensus = FALSE,
    augment.path = FALSE
  )

  # Adjusted Multiresolution Rand Index (AMRI)
  ks.flat <-  apply(
    out$labelmat.flat,
    2,
    FUN = function(x)
      length(unique(x))
  )
  ks.mrtree <-  apply(
    out$labelmat.mrtree,
    2,
    FUN = function(x)
      length(unique(x))
  )
  amri.flat <- sapply(1:ncol(out$labelmat.flat), function(i)
    AMRI(out$labelmat.flat[, i], srt$seurat_clusters)$amri)
  amri.flat <- aggregate(amri.flat, by = list(k = ks.flat), FUN = mean)
  amri.recon <- sapply(1:ncol(out$labelmat.mrtree), function(i)
    AMRI(out$labelmat.mrtree[, i], srt$seurat_clusters)$amri)

  df <- rbind(
    data.frame(
      k = amri.flat$k,
      amri = amri.flat$x,
      method = 'Seurat flat'
    ),
    data.frame(k = ks.mrtree, amri = amri.recon, method = 'MRtree')
  )

  stab.out <- stability_plot(out)

  tmp.ari <- stab.out$df |>
    as_tibble() |>
    filter(ari != 1) |>
    top_n(n = 2, wt = ari) |>
    purrr::pluck(2)
  tmp.ari <- tmp.ari[1] - tmp.ari[2]
  if (tmp.ari < 0.05) {
    resK <-
      stab.out$df |>
      as_tibble() |>
      filter(ari != 1) |>
      top_n(n = 2, wt = ari) |>
      purrr::pluck(1)
    resK <- resK[2]
  } else {
    resK <-
      stab.out$df |>
      as_tibble() |>
      filter(ari != 1) |>
      top_n(n = 1, wt = ari) |>
      purrr::pluck(1)
  }

  kable_material(
    kable(
      table(
        out$labelmat.mrtree[, which.min(
          abs(as.integer(
            str_remove(dimnames(
              out$labelmat.mrtree)[[2]], "K"
            )
          ) - resK)
        )]
      ),
      "html"),
    bootstrap_options = "striped",
    position = "left",
    font_size = 7
  )

  srt$k_tree <- out$labelmat.mrtree[, which.min(
    abs(as.integer(
      str_remove(dimnames(
        out$labelmat.mrtree)[[2]], "K"
      )
    ) - resK)
  )]

  Idents(srt) <- "k_tree"
  if (length(unique(srt$k_tree)) > 1) {
  srt.markers.lr <-
    FindAllMarkers(
      srt,
      assay = "SCT",
      latent.vars = c("percent_mt"),
      only.pos = TRUE,
      min.diff.pct = 0.15,
      logfc.threshold = 0.5,
      test.use = "LR")

  if (length(unique(srt.markers.lr$cluster)) > 1) {
    write_csv(
      srt.markers.lr,
      here(tables_dir,
           sprintf('%s_all-mrk_logreg-sct-intern.csv',
                   unique(srt$Batch_ID))))

    srt.markers.lr %>%
      group_by(cluster) %>%
      top_n(n = 10, wt = avg_log2FC) -> top10

    kable_material(
      kable(
        top10,
        "html"),
      bootstrap_options = "striped",
      position = "left",
      font_size = 7
    )
  }
  }
  plan(sequential)
  return(srt)
}
