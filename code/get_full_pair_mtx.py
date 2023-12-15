#!/usr/bin/env python
# coding: utf-8

import os
import random

import anndata as ad
import matplotlib
import numpy as np
import pacmap
import pandas as pd
import scanpy as sc
import scanpy.experimental as scexp
import scanpy.external as scext

PLOTS_DIR = os.path.join("output/figures/paired_integrations-full/")

matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica"]
matplotlib.rcParams["figure.max_open_warning"] = 20000

reseed = 42
random.seed(reseed)
np.random.seed(reseed)

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


# Load data:

sample1 = snakemake.params["fullpairintegr"]["whole_hypothalamus"]
sample2 = snakemake.params["fullpairintegr"]["subregions"]
cb_fpr = snakemake.params["fullpairintegr"]["fpr"]

## IN:
data_path1 = os.path.join(f"{sample1}/data/class_cello/", f"{sample1}-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad")
data_path2 = os.path.join(f"{sample2}/data/class_cello/", f"{sample2}-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad")


## OUT:
h5ad_out_path = snakemake.output["h5ad_pair"]
genes_table_out = snakemake.output["anchor_genes"]

adata1 = sc.read_h5ad(data_path1)
print(adata1)
sc.pp.normalize_total(adata1, target_sum=1e4)
sc.pp.log1p(adata1)
scexp.pp.highly_variable_genes(adata1, n_top_genes=10000, batch_key=["orig.ident"])
adata1.raw = adata1
adata1 = adata1[:, adata1.var.highly_variable]
print(adata1)

adata2 = sc.read_h5ad(data_path2)
print(adata2)
sc.pp.normalize_total(adata2, target_sum=1e4)
sc.pp.log1p(adata2)
scexp.pp.highly_variable_genes(adata2, n_top_genes=10000, batch_key=["orig.ident"])
adata2.raw = adata2
adata2 = adata2[:, adata2.var.highly_variable]
print(adata2)

# merge pair and select only controls
adata = ad.concat([adata1, adata2], join='inner')
adata = adata[adata.obs["condit"] == 0]
adata = adata[:, ~adata.var_names.str.match(r"(^Hla-|^Ig[hjkl]|^Rna|^mt-|^Rp[sl]|^Hb[^(p)]|^Gm)")]


# derive shared highly variable genes and run PCA on top
scexp.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key=["orig.ident"])
adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['nCount_RNA', 'percent_mito_ribo'])
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')

print("Show genes that are highly variable across all samples")
var_genes = adata.var['highly_variable_rank'][adata.var['highly_variable_intersection']]
print(var_genes)
# save highly variable genes
pd.DataFrame(adata.var).to_csv(genes_table_out, sep="\t", header=True)

# derive shared latent space representation for future analysis
integration_keys = ['orig.ident', 'project', 'model', 'tech', 'expbtch']
integration_keys = [key for key in integration_keys if adata.obs[key].nunique() > 1]
scext.pp.harmony_integrate(
    adata=adata,
    key=integration_keys,
    max_iter_harmony=30,
    adjusted_basis='X_pca_harmony',
    random_state=reseed)

# visualise metadata in integrated latent space
embedding = pacmap.PaCMAP(
    n_components=2,
    num_iters=1000,
    n_neighbors=None,
    MN_ratio=0.5,
    FP_ratio=2.0,
    apply_pca=True
)
adata.obsm["X_pacmap"] = embedding.fit_transform(
    adata.obsm["X_pca_harmony"], init="pca")

save_paired = f"-paired_mtx_{sample1}_and_{sample2}-full-astrocytes_datasets_fpr_{cb_fpr}.pdf"

plot_meta_keys = ['project', 'model', 'tech', 'region', 'sex',  'stage',  'libname', 'expbtch', 'condit']
plot_meta_keys = [key for key in plot_meta_keys if adata.obs[key].nunique() > 1]
sc.pl.embedding(
    adata,
    basis="X_pacmap",
    color=plot_meta_keys,
    alpha=0.5,
    save=save_paired,
)

sc.pl.embedding(
    adata,
    basis="X_pacmap",
    color=['project'],
    palette={
        "lutomska2022-Arc": "magenta",
        "pool2022_MnPO": "magenta",
        "liu2022-VMHvl": "magenta",
        "rupp2021_MBH": "magenta",
        "affinati2021_VMH": "magenta",
        "morris2021_SCN": "magenta",
        "lopez2021_PVN": "magenta",
        "mickelsen2020_VPH": "magenta",
        "deng2020_Arc": "magenta",
        "wen2020_SCN": "magenta",
        "mickelsen2019_LHA": "magenta",
        "moffitt2018_POA": "magenta",
        "zeisel2018_Hypoth-brain": "black",
        "romanov2020_Hypoth-dev": "black",
        "kim2020_Hypoth-dev": "black",
        "hajdarovic2022_Hypoth": "black",
    },
    alpha=0.5,
    title=None,
    save="-project" + save_paired,
)



# save results
if not os.path.exists(os.path.dirname(h5ad_out_path)):
    os.mkdir(os.path.dirname(h5ad_out_path))
adata.write(h5ad_out_path)

# save results
if not os.path.exists(os.path.dirname(h5ad_out_path)):
    os.mkdir(os.path.dirname(h5ad_out_path))
adata.write(h5ad_out_path)