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
from nancorrmp.nancorrmp import NaNCorrMp
from natsort import natsorted
from sklearn.svm import SVC

######################
##### parameters #####
######################

PLOTS_DIR = os.path.join("output/figures/paired_integrations-wo_signature/")

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
    transparent=True,
    color_map="gnuplot_r",
)
sc.settings.autoshow = False
sc.settings.autosave = True

sample1 = snakemake.params["sample1"]
sample2 = snakemake.params["sample2"]
cb_fpr = snakemake.params["res"]
n_processes = int(snakemake.threads)

#####################
##### functions #####
#####################

def get_top_abs_correlations(df):
    """Get the unstack table of pairs of correlation matrix"""
    return df.abs().unstack().sort_values(ascending=False).rename_axis(["Gene1", "Gene2"], axis=0)


def remove_irrelevant_top_abs_correlations(corr_df, signature_genes, adata):
    """Indicate highly correlated genes with regard to the signature"""
    signature_genes = signature_genes[signature_genes.isin(adata.var_names)]
    cor_targets = corr_df.loc[signature_genes.tolist()]
    genes_table = cor_targets[cor_targets > 0.5].reset_index()
    genes = set([g for g in genes_table['Gene1']] + [g for g in genes_table['Gene2']])
    return list(genes)


######################
##### Load data ######
######################

## IN:
housekeeping = "data/housekeeping_mouse.tsv"
data_path = snakemake.input["h5ad_pair"]
genes_table = snakemake.input["aggregated_shared_genes"]
data_path1 = os.path.join(
    f"{sample1}/data/class_cello/",
    f"{sample1}-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad",
)
data_path2 = os.path.join(
    f"{sample2}/data/class_cello/",
    f"{sample2}-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad",
)


## OUT:
genes_corr_p_value = snakemake.output["tables_genes_corr_p_value"]
genes_corr = snakemake.output["tables_genes_corr"]
irrcorr = snakemake.output["tables_genes_irrelevant"]
h5ad_out_path_100 = f"data/paired_integrations-wo_signature/paired_mtx-{sample1}_and_{sample2}-100-astrocytes_datasets_{cb_fpr}.h5ad"
h5ad_out_path_250 = f"data/paired_integrations-wo_signature/paired_mtx-{sample1}_and_{sample2}-250-astrocytes_datasets_{cb_fpr}.h5ad"
h5ad_out_path_500 = f"data/paired_integrations-wo_signature/paired_mtx-{sample1}_and_{sample2}-500-astrocytes_datasets_{cb_fpr}.h5ad"

sign_genes = pd.read_table(genes_table)
hk_genes1 = []
with open(housekeeping) as file:
    while hk_genes := file.readline():
        hk_genes1.append(hk_genes.rstrip())


adata1 = sc.read_h5ad(data_path1)
print(adata1)
sc.pp.normalize_total(adata1, target_sum=1e4)
sc.pp.log1p(adata1)

adata2 = sc.read_h5ad(data_path2)
print(adata2)
sc.pp.normalize_total(adata2, target_sum=1e4)
sc.pp.log1p(adata2)


# merge pair and select only controls
adata = ad.concat([adata1, adata2], join="inner")
adata = adata[adata.obs["condit"] == 0]
adata = adata[
    :, ~adata.var_names.str.match(r"(^Hla-|^Ig[hjkl]|^Rna|^mt-|^Rp[sl]|^Hb[^(p)]|^Gm)")
]
adata = adata[:, ~adata.var_names.isin(hk_genes1)]
adata_tmp = adata.copy()
sc.pp.regress_out(adata_tmp, ["nCount_RNA", "percent_mito_ribo"])
sc.pp.scale(adata_tmp, max_value=10)
X = pd.DataFrame(adata_tmp.X, columns=adata_tmp.var_names, index=adata_tmp.obs_names)


corr, p_value = NaNCorrMp.calculate_with_p_value(X, n_jobs=n_processes)
corr.to_csv(
    genes_corr,
    sep="\t",
    header=True,
)

p_value.to_csv(
    genes_corr_p_value,
    sep="\t",
    header=True,
)

# Add absolute correlation table
abs_correlation_table = get_top_abs_correlations(corr)
print(abs_correlation_table)


sign_genes_100 = sign_genes.sort_values(["Score"], ascending=True).head(100)
irrcorr_100 = remove_irrelevant_top_abs_correlations(
    corr_df=abs_correlation_table,
    signature_genes=sign_genes_100.Name,
    adata=adata,
)
with open(irrcorr[0], "w") as outfile:
    outfile.write("\n".join(str(i) for i in irrcorr_100))

adata_100 = adata[:, ~adata.var_names.isin(sign_genes_100.Name)]

adata_100_train = adata_100[adata.obs["train"], :]
adata_100_train = adata_100_train[:, ~adata_100_train.var_names.isin(irrcorr_100)]
scexp.pp.highly_variable_genes(
    adata_100_train, n_top_genes=2000, batch_key=["orig.ident"]
)
adata_100_train.raw = adata_100_train
adata_100_train = adata_100_train[:, adata_100_train.var.highly_variable]
sc.pp.regress_out(adata_100_train, ["nCount_RNA", "percent_mito_ribo"])
sc.pp.scale(adata_100_train, max_value=10)
sc.tl.pca(adata_100_train, svd_solver="arpack")

scexp.pp.highly_variable_genes(adata_100, n_top_genes=5000, batch_key=["orig.ident"])
adata_100.var.highly_variable[adata_100.var.highly_variable] = ~adata_100.var_names[
    adata_100.var.highly_variable
].isin(irrcorr_100)
adata_100.raw = adata_100
sc.pp.regress_out(adata_100, ["nCount_RNA", "percent_mito_ribo"])
sc.pp.scale(adata_100, max_value=10)
sc.tl.pca(adata_100, svd_solver="arpack")


sign_genes_250 = sign_genes.sort_values(["Score"], ascending=True).head(250)
irrcorr_250 = remove_irrelevant_top_abs_correlations(
    corr_df=abs_correlation_table,
    signature_genes=sign_genes_250.Name,
    adata=adata,
)
with open(irrcorr[1], "w") as outfile:
    outfile.write("\n".join(str(i) for i in irrcorr_250))

adata_250 = adata[:, ~adata.var_names.isin(sign_genes_250.Name)]

adata_250_train = adata_250[adata.obs["train"], :]
adata_250_train = adata_250_train[:, ~adata_250_train.var_names.isin(irrcorr_250)]
scexp.pp.highly_variable_genes(
    adata_250_train, n_top_genes=2000, batch_key=["orig.ident"]
)
adata_250_train.raw = adata_250_train
adata_250_train = adata_250_train[:, adata_250_train.var.highly_variable]
sc.pp.regress_out(adata_250_train, ["nCount_RNA", "percent_mito_ribo"])
sc.pp.scale(adata_250_train, max_value=10)
sc.tl.pca(adata_250_train, svd_solver="arpack")

scexp.pp.highly_variable_genes(adata_250, n_top_genes=5000, batch_key=["orig.ident"])
adata_250.var.highly_variable[adata_250.var.highly_variable] = ~adata_250.var_names[
    adata_250.var.highly_variable
].isin(irrcorr_250)
adata_250.raw = adata_250
sc.pp.regress_out(adata_250, ["nCount_RNA", "percent_mito_ribo"])
sc.pp.scale(adata_250, max_value=10)
sc.tl.pca(adata_250, svd_solver="arpack")


sign_genes_500 = sign_genes.sort_values(["Score"], ascending=True).head(500)
irrcorr_500 = remove_irrelevant_top_abs_correlations(
    corr_df=abs_correlation_table,
    signature_genes=sign_genes_500.Name,
    adata=adata,
)
with open(irrcorr[2], "w") as outfile:
    outfile.write("\n".join(str(i) for i in irrcorr_500))

adata_500 = adata[:, ~adata.var_names.isin(sign_genes_500.Name)]

adata_500_train = adata_500[adata.obs["train"], :]
adata_500_train = adata_500_train[:, ~adata_500_train.var_names.isin(irrcorr_500)]
scexp.pp.highly_variable_genes(
    adata_500_train, n_top_genes=2000, batch_key=["orig.ident"]
)
adata_500_train.raw = adata_500_train
adata_500_train = adata_500_train[:, adata_500_train.var.highly_variable]
sc.pp.regress_out(adata_500_train, ["nCount_RNA", "percent_mito_ribo"])
sc.pp.scale(adata_500_train, max_value=10)
sc.tl.pca(adata_500_train, svd_solver="arpack")

scexp.pp.highly_variable_genes(adata_500, n_top_genes=5000, batch_key=["orig.ident"])
adata_500.var.highly_variable[adata_500.var.highly_variable] = ~adata_500.var_names[
    adata_500.var.highly_variable
].isin(irrcorr_500)
adata_500.raw = adata_500
sc.pp.regress_out(adata_500, ["nCount_RNA", "percent_mito_ribo"])
sc.pp.scale(adata_500, max_value=10)
sc.tl.pca(adata_500, svd_solver="arpack")

# derive shared latent space representation for future analysis
integration_keys = ["orig.ident", "project", "model", "tech", "expbtch"]
integration_keys = [key for key in integration_keys if adata.obs[key].nunique() > 1]


scext.pp.harmony_integrate(
    adata=adata_100_train,
    key=integration_keys,
    adjusted_basis="X_pca_harmony",
    max_iter_harmony=20,
    random_state=reseed,
)
sc.pp.neighbors(
    adata_100_train,
    n_neighbors=10,
    use_rep="X_pca_harmony",
    metric="cosine",
    knn=True,
    random_state=reseed,
)
sc.tl.leiden(adata_100_train, random_state=reseed)
adata_100_train.obs["leiden"] = pd.Categorical(
    values=adata_100_train.obs["leiden"].astype("U"),
    categories=natsorted(
        map(str, np.unique(adata_100_train.obs["leiden"].astype(str)))
    ),
)

scext.pp.harmony_integrate(
    adata=adata_250_train,
    key=integration_keys,
    adjusted_basis="X_pca_harmony",
    max_iter_harmony=20,
    random_state=reseed,
)
sc.pp.neighbors(
    adata_250_train,
    n_neighbors=10,
    use_rep="X_pca_harmony",
    metric="cosine",
    knn=True,
    random_state=reseed,
)
sc.tl.leiden(adata_250_train, random_state=reseed)
adata_250_train.obs["leiden"] = pd.Categorical(
    values=adata_250_train.obs["leiden"].astype("U"),
    categories=natsorted(
        map(str, np.unique(adata_250_train.obs["leiden"].astype(str)))
    ),
)

scext.pp.harmony_integrate(
    adata=adata_500_train,
    key=integration_keys,
    adjusted_basis="X_pca_harmony",
    max_iter_harmony=20,
    random_state=reseed,
)
sc.pp.neighbors(
    adata_500_train,
    n_neighbors=10,
    use_rep="X_pca_harmony",
    metric="cosine",
    knn=True,
    random_state=reseed,
)
sc.tl.leiden(adata_500_train, random_state=reseed)
adata_500_train.obs["leiden"] = pd.Categorical(
    values=adata_500_train.obs["leiden"].astype("U"),
    categories=natsorted(
        map(str, np.unique(adata_500_train.obs["leiden"].astype(str)))
    ),
)


scext.pp.harmony_integrate(
    adata=adata_100,
    key=integration_keys,
    adjusted_basis="X_pca_harmony",
    max_iter_harmony=30,
    random_state=reseed,
)
sc.pp.neighbors(
    adata_100,
    n_neighbors=20,
    use_rep="X_pca_harmony",
    metric="cosine",
    knn=True,
    random_state=reseed,
)

scext.pp.harmony_integrate(
    adata=adata_250,
    key=integration_keys,
    adjusted_basis="X_pca_harmony",
    max_iter_harmony=30,
    random_state=reseed,
)
sc.pp.neighbors(
    adata_250,
    n_neighbors=20,
    use_rep="X_pca_harmony",
    metric="cosine",
    knn=True,
    random_state=reseed,
)

scext.pp.harmony_integrate(
    adata=adata_500,
    key=integration_keys,
    adjusted_basis="X_pca_harmony",
    max_iter_harmony=30,
    random_state=reseed,
)
sc.pp.neighbors(
    adata_500,
    n_neighbors=20,
    use_rep="X_pca_harmony",
    metric="cosine",
    knn=True,
    random_state=reseed,
)


X_train, X_test = (
    adata_100[adata_100.obs["train"]].obsm["X_pca_harmony"],
    adata_100[adata_100.obs["test"]].obsm["X_pca_harmony"],
)
lbl_100 = adata_100_train.obs["leiden"].astype(int)
# Fit linear separator using X_train
svm = SVC(kernel="linear", C=100)
svm.fit(X_train, lbl_100)

# Generate labels for X_test
predictions = svm.predict(X_test)
adata_100.obs["leiden"] = adata_100.obs["condit"].astype(int)
adata_100.obs["leiden"][adata.obs["train"]] = lbl_100
adata_100.obs["leiden"][adata.obs["test"]] = predictions
adata_100.obs["leiden"] = pd.Categorical(
    values=adata_100.obs["leiden"].astype("U"),
    categories=natsorted(map(str, np.unique(adata_100.obs["leiden"].astype(str)))),
)


X_train, X_test = (
    adata_250[adata_250.obs["train"]].obsm["X_pca_harmony"],
    adata_250[adata_250.obs["test"]].obsm["X_pca_harmony"],
)
lbl_250 = adata_250_train.obs["leiden"].astype(int)
# Fit linear separator using X_train
svm = SVC(kernel="linear", C=100)
svm.fit(X_train, lbl_250)

# Generate labels for X_test
predictions = svm.predict(X_test)
adata_250.obs["leiden"] = adata_250.obs["condit"].astype(int)
adata_250.obs["leiden"][adata.obs["train"]] = lbl_250
adata_250.obs["leiden"][adata.obs["test"]] = predictions
adata_250.obs["leiden"] = pd.Categorical(
    values=adata_250.obs["leiden"].astype("U"),
    categories=natsorted(map(str, np.unique(adata_250.obs["leiden"].astype(str)))),
)


X_train, X_test = (
    adata_500[adata_500.obs["train"]].obsm["X_pca_harmony"],
    adata_500[adata_500.obs["test"]].obsm["X_pca_harmony"],
)
lbl_500 = adata_500_train.obs["leiden"].astype(int)
# Fit linear separator using X_train
svm = SVC(kernel="linear", C=100)
svm.fit(X_train, lbl_500)

# Generate labels for X_test
predictions = svm.predict(X_test)
adata_500.obs["leiden"] = adata_500.obs["condit"].astype(int)
adata_500.obs["leiden"][adata.obs["train"]] = lbl_500
adata_500.obs["leiden"][adata.obs["test"]] = predictions
adata_500.obs["leiden"] = pd.Categorical(
    values=adata_500.obs["leiden"].astype("U"),
    categories=natsorted(map(str, np.unique(adata_500.obs["leiden"].astype(str)))),
)
# TODO: fix neighbors graphs and separate embeddings for each substraction
# visualise metadata in integrated latent space
plot_meta_keys = [
    "project",
    "model",
    "tech",
    "region",
    "sex",
    "stage",
    "libname",
    "expbtch",
    "condit",
]
plot_meta_keys = [key for key in plot_meta_keys if adata.obs[key].nunique() > 1]

embedding1 = pacmap.PaCMAP(
    n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0,
    num_iters=1000, apply_pca=True
)
adata_100.obsm["X_pacmap"] = embedding1.fit_transform(
    adata_100.obsm["X_pca_harmony"],
    init="pca"
)

save_paired_100 = (
    f"-paired_mtx-{sample1}_and_{sample2}-100-astrocytes_datasets_{cb_fpr}.pdf"
)

sc.pl.embedding(
    adata_100,
    basis="X_pacmap",
    color=plot_meta_keys,
    title=f"PaCMAP: paired integration wo 100, {sample1} + {sample2} (amb.FPR={cb_fpr})",
    alpha=0.5,
    save=save_paired_100,
)

sc.pl.embedding(
    adata_100,
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
        "zeisel2018_Hypoth": "black",
        "romanov2020_Hypoth-dev": "black",
        "kim2020_Hypoth-dev": "black",
        "hajdarovic2022_Hypoth": "black",
    },
    alpha=0.5,
    title=None,
    save="-project" + save_paired_100,
)


embedding2 = pacmap.PaCMAP(
    n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0,
    num_iters=1000, apply_pca=True
)
adata_250.obsm["X_pacmap"] = embedding2.fit_transform(
    adata_250.obsm["X_pca_harmony"],
    init="pca"
)

save_paired_250 = (
    f"-paired_mtx-{sample1}_and_{sample2}-250-astrocytes_datasets_{cb_fpr}.pdf"
)

sc.pl.embedding(
    adata_250,
    basis="X_pacmap",
    color=plot_meta_keys,
    title=f"PaCMAP: paired integration wo 250, {sample1} + {sample2} (amb.FPR={cb_fpr})",
    alpha=0.5,
    save=save_paired_250,
)

sc.pl.embedding(
    adata_250,
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
        "zeisel2018_Hypoth": "black",
        "romanov2020_Hypoth-dev": "black",
        "kim2020_Hypoth-dev": "black",
        "hajdarovic2022_Hypoth": "black",
    },
    alpha=0.5,
    title=None,
    save="-project" + save_paired_250,
)


embedding3 = pacmap.PaCMAP(
    n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0,
    num_iters=1000, apply_pca=True
)
adata_500.obsm["X_pacmap"] = embedding3.fit_transform(
    adata_500.obsm["X_pca_harmony"],
    init="pca"
)

save_paired_500 = (
    f"-paired_mtx-{sample1}_and_{sample2}-500-astrocytes_datasets_{cb_fpr}.pdf"
)

sc.pl.embedding(
    adata_500,
    basis="X_pacmap",
    color=plot_meta_keys,
    title=f"PaCMAP: paired integration wo 500, {sample1} + {sample2} (amb.FPR={cb_fpr})",
    alpha=0.5,
    save=save_paired_500,
)

sc.pl.embedding(
    adata_500,
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
        "zeisel2018_Hypoth": "black",
        "romanov2020_Hypoth-dev": "black",
        "kim2020_Hypoth-dev": "black",
        "hajdarovic2022_Hypoth": "black",
    },
    alpha=0.5,
    title=None,
    save="-project" + save_paired_500,
)

# save results
if not os.path.exists(os.path.dirname(h5ad_out_path_100)):
    os.mkdir(os.path.dirname(h5ad_out_path_100))
adata_100.write(h5ad_out_path_100)
adata_250.write(h5ad_out_path_250)
adata_500.write(h5ad_out_path_500)
# save results
if not os.path.exists(os.path.dirname(h5ad_out_path_100)):
    os.mkdir(os.path.dirname(h5ad_out_path_100))
adata_100.write(h5ad_out_path_100)
adata_250.write(h5ad_out_path_250)
adata_500.write(h5ad_out_path_500)
