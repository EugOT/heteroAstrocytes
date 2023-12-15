#!/usr/bin/env python
# coding: utf-8

import os
import random
import scanpy as sc, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
from matplotlib import rcParams

matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica"]
matplotlib.rcParams["figure.max_open_warning"] = 20000
PLOTS_DIR = os.path.dirname(snakemake.output["top25_figures_logreg"])
reseed = 42
random.seed(reseed)
np.random.seed(reseed)
sc.settings.figdir = PLOTS_DIR
sc.settings.set_figure_params(
    dpi=120, dpi_save=600, vector_friendly=True, format="pdf", transparent=True
)
sc.settings.autoshow = False
sc.settings.autosave = True
sc.set_figure_params(dpi=150, color_map = 'viridis_r')
sc.settings.verbosity = 1
sc.logging.print_header()

metric = snakemake.params["metric"]
connectivity_model = snakemake.params["connectivity_model"]
k = int(snakemake.params["k"])
signature = snakemake.params["substr_sign"]
cb_fpr = snakemake.params["res"]
subregions = snakemake.params["subregions"]

adata = sc.read_h5ad(snakemake.input["h5ad"])
adata.obs.clusters = adata.obs.clusters.astype(str)
adata.obs["clusters"] = pd.Categorical(adata.obs["clusters"])
sc.tl.rank_genes_groups(adata, 'clusters', method='logreg')

save_plot = f"-logreg-{subregions}-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save=save_plot)

# with rc_context({'figure.figsize': (9, 1.5)}):
#     sc.pl.rank_genes_groups_violin(adata, n_genes=20, jitter=True, save=save_plot)

# sc.pl.rank_genes_groups_dotplot(adata, n_genes=10, cmap='bwr', save=save_plot)

# sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, use_raw=False, swap_axes=True, vmin=-3, vmax=3, cmap='bwr', layer='scaled', figsize=(10,7), show=False, save=save_plot)

# sc.pl.rank_genes_groups_tracksplot(adata, n_genes=10, save=save_plot)
