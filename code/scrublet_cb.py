import tables
import anndata
import pacmap
import os
import random
import sys
import time
import traceback
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
import scrublet as scr
import scipy.sparse as sp
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
PLOTS_DIR = os.path.join(snakemake.params["plots"])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.max_open_warning'] = 300

reseed = 42
random.seed(reseed)
np.random.seed(reseed)
n_cores=snakemake.threads

sc.settings.verbosity = 2  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.figdir = PLOTS_DIR
sc.settings.set_figure_params(
    dpi=120,
    dpi_save=600,
    vector_friendly=True,
    format="pdf",
    transparent=True)
sc.settings.autoshow = False
sc.settings.autosave = True
sc.logging.print_versions()


def adata_to_scrublet(data_path, scr_out_path, h5ad_out_path, expected_dblt, sample_name):
    # load the data
    adata = anndata.read(data_path)
    adata.uns["name"] = sample_name
    adata.var_names_make_unique()

    scrub = scr.Scrublet(adata.X, expected_doublet_rate = expected_dblt)
    adata.obs['doublet_score'], adata.obs['predicted_doublets'] =     scrub.scrub_doublets(min_counts=2, min_cells=3,     min_gene_variability_pctl=85, n_prin_comps=30)
    embedding = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=None,
        MN_ratio=0.5,
        FP_ratio=2.0)
    adata.obsm['X_pacmap'] = embedding.fit_transform(
        adata.X.toarray(),
        init="pca")
    sc.pl.embedding(
        adata,
        basis='X_pacmap',
        color='doublet_score',
        title='PaCMAP: Doublets score derived using Scrublet in {sample}'.format(sample=adata.uns["name"]),
        save="_doublet-score_{sample}.pdf".format(sample=adata.uns["name"]))
    # Save results:
    adata.write(h5ad_out_path)
    pd.DataFrame(adata.obs).to_csv(scr_out_path, sep = '\t', header = True) # scrublet_calls.tsv


adata_to_scrublet(data_path=snakemake.input["filt_h5ad"], scr_out_path=snakemake.output["scrublet_calls"], h5ad_out_path=snakemake.output["h5ad"], expected_dblt=snakemake.params["expected_dblt"], sample_name=snakemake.params["sample_run_name"])