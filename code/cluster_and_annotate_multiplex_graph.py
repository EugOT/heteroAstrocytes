#!/usr/bin/env python
# coding: utf-8

import os
import random
import warnings
import time
import copy
import heapq
import pickle
import itertools
import sklearn
import scanpy as sc, scanpy.external as scext, scanpy.experimental as scexp, anndata as ad, numpy as np, pandas as pd, matplotlib.pyplot as plt, leidenalg as la, igraph as ig, networkx as nx
import scipy
from scipy.stats import ttest_ind
from truncated_normal import truncated_normal as tn
from sklearn.svm import SVC
from natsort import natsorted
from sklearn.utils import check_random_state
import joblib
import matplotlib
import pacmap
import umap
from umap.umap_ import (
    find_ab_params,
    nearest_neighbors,
    fuzzy_simplicial_set,
    simplicial_set_embedding,
    dist,
)


#################
##### genes #####
#################

npr = [
    "Adcyap1r1",
    "Avpr1a",
    "Cckar",
    "Cckbr",
    "Cntfr",
    "Crhr1",
    "Crhr2",
    "Esr1",
    "Galr1",
    "Galr2",
    "Galr3",
    "Ghr",
    "Ghrhr",
    "Gpr83",
    "Grpr",
    "Hcrtr1",
    "Hcrtr2",
    "Kiss1r",
    "Mc1r",
    "Mc3r",
    "Mc4r",
    "Mchr1",
    "Nmbr",
    "Nmur1",
    "Nmur2",
    "Npffr1",
    "Npffr2",
    "Npy1r",
    "Npy2r",
    "Npy5r",
    "Ntrk2",
    "Ntsr2",
    "Oprd1",
    "Oprk1",
    "Oprl1",
    "Oprm1",
    "Oxtr",
    "Prlhr",
    "Prlr",
    "Prokr2",
    "Qrfpr",
    "Sstr1",
    "Sstr2",
    "Sstr3",
    "Tacr1",
    "Tacr3",
    "Trhr",
    "Vipr1",
    "Vipr2",
]

npep = [
    "Adcyap1",
    "Agrp",
    "Avp",
    "Bdnf",
    "Cartpt",
    "Cck",
    "Cntf",
    "Crh",
    "Gal",
    "Ghrh",
    "Grp",
    "Hcrt",
    "Kiss1",
    "Nmb",
    "Nms",
    "Nmu",
    "Npvf",
    "Npw",
    "Npy",
    "Nts",
    "Oxt",
    "Pdyn",
    "Penk",
    "Pmch",
    "Pnoc",
    "Pomc",
    "Qrfp",
    "Reln",
    "Rxfp1",
    "Sst",
    "Tac1",
    "Tac2",
    "Trh",
]

genes_embed = [
    "Abcd1",
    "Abcd2",
    "Abcd3",
    "Acaa1",
    "Acaa2",
    "Acox1",
    "Agt",
    "Aldh1a1",
    "Aldh1l1",
    "Aldoc",
    "Apoe",
    "Aqp4",
    "Caf4",
    "Ckb",
    "Cnr1",
    "Cnr2",
    "Cst3",
    "Dagla",
    "Daglb",
    "Decr2",
    "Dnm1",
    "Drp1",
    "Ech1",
    "Egfr",
    "Eno1",
    "Faah",
    "Fgfr3",
    "Fis1",
    "Fos",
    "Fth1",
    "Ftl1",
    "Gfap",
    "Gja1",
    "Gli1",
    "Glul",
    "Hacd2",
    "Hadhb",
    "Hepacam",
    "Hif1",
    "Htra1",
    "Lcat",
    "Lpcat3",
    "Lxn",
    "Mdv1",
    "Mfn1",
    "Mfn2",
    "Mgll",
    "Mief1",
    "Napepld",
    "Ndrg2",
    "Nfia",
    "Opa1",
    "Otp",
    "Pex1",
    "Pex10",
    "Pex12",
    "Pex13",
    "Pex14",
    "Pex16",
    "Pex2",
    "Pex26",
    "Pex3",
    "Pex6",
    "Pla2g7",
    "Plcb1",
    "Pygb",
    "S100a6",
    "S100b",
    "Scd2",
    "Sgcd",
    "Slc1a2",
    "Slc1a3",
    "Slc38a1",
    "Slc6a11",
    "Slit1",
    "Slit2",
    "Sox9",
    "Tafa1",
    "Tkt",
    "Trpv1",
]

nmr = [
    "Gria1", "Gria2", "Gria3", "Gria4", # iGlu AMPA receptors
    "Grid1", "Grid2", # iGlu delta receptors
    "Grik1", "Grik2", "Grik3", "Grik4", "Grik5", # iGlu kainate receptors
    "Grin1", "Grin2a", "Grin2b", "Grin2c", "Grin2d", "Grin3a", "Grin3b", # iGlu NMDA receptors
    "Grm1", "Grm5", # mGluRs 1
    "Grm2", "Grm3", # mGluRs 2
    "Grm4", "Grm6", "Grm7", "Grm8",# mGluRs 3
    "Gabra1", "Gabra2", "Gabra3", "Gabra4", "Gabra5", "Gabra6",
    "Gabrb1", "Gabrb2", "Gabrb3",
    "Gabrg1", "Gabrg2", "Gabrg3",
    "Gabrd", "Gabre", "Gabrp", "Gabrq",
    "Gabrr1", "Gabrr2", "Gabrr3",
    "Gabbr1", "Gabbr2",
    "Drd1", "Drd2", "Drd3", "Drd4", "Drd5",
    "Htr1a", "Htr1b", "Htr1d", "Htr1f", "Htr2a", "Htr2b", "Htr2c", "Htr3a", "Htr3b", "Htr4", "Htr5a", "Htr5b", "Htr6", "Htr7", "Gnai1", "Gnai2", "Gnai3", "Gnao1", "Gnaz",
]


#####################
##### functions #####
#####################


# Calculate min spanning tree
def min_spanning_tree(knn_indices, knn_dists, n_neighbors, threshold):
    rows = np.zeros(knn_indices.shape[0] * n_neighbors, dtype=np.int32)
    cols = np.zeros(knn_indices.shape[0] * n_neighbors, dtype=np.int32)
    vals = np.zeros(knn_indices.shape[0] * n_neighbors, dtype=np.float32)

    pos = 0
    for i, indices in enumerate(knn_indices):
        for j, index in enumerate(indices[:threshold]):
            if index == -1:
                continue
            rows[pos] = i
            cols[pos] = index
            vals[pos] = knn_dists[i][j]
            pos += 1

    matrix = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(knn_indices.shape[0], knn_indices.shape[0])
    )
    Tcsr = scipy.sparse.csgraph.minimum_spanning_tree(matrix)

    Tcsr = scipy.sparse.coo_matrix(Tcsr)
    weights_tuples = zip(Tcsr.row, Tcsr.col, Tcsr.data)

    sorted_weights_tuples = sorted(weights_tuples, key=lambda tup: tup[2])

    return sorted_weights_tuples


def create_connected_graph(
    mutual_nn, total_mutual_nn, knn_indices, knn_dists, n_neighbors, connectivity
):
    connected_mnn = copy.deepcopy(mutual_nn)

    if connectivity == "nearest":
        for i in range(len(knn_indices)):
            if len(mutual_nn[i]) == 0:
                first_nn = knn_indices[i][1]
                if first_nn != -1:
                    connected_mnn[i].add(first_nn)
                    connected_mnn[first_nn].add(i)
                    total_mutual_nn += 1
        return connected_mnn

    # Create graph for mutual NN
    rows = np.zeros(total_mutual_nn, dtype=np.int32)
    cols = np.zeros(total_mutual_nn, dtype=np.int32)
    vals = np.zeros(total_mutual_nn, dtype=np.float32)
    pos = 0
    for i in connected_mnn:
        for j in connected_mnn[i]:
            rows[pos] = i
            cols[pos] = j
            vals[pos] = 1
            pos += 1
    graph = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(knn_indices.shape[0], knn_indices.shape[0])
    )

    # Find number of connected components
    n_components, labels = scipy.sparse.csgraph.connected_components(
        csgraph=graph, directed=True, return_labels=True, connection="strong"
    )
    print(n_components)
    label_mapping = {i: [] for i in range(n_components)}

    for index, component in enumerate(labels):
        label_mapping[component].append(index)

    # Find the min spanning tree with KNN
    sorted_weights_tuples = min_spanning_tree(
        knn_indices, knn_dists, n_neighbors, n_neighbors
    )

    # Add edges until graph is connected
    for pos, (i, j, v) in enumerate(sorted_weights_tuples):
        if connectivity == "full_tree":
            connected_mnn[i].add(j)
            connected_mnn[j].add(i)

        elif connectivity == "min_tree" and labels[i] != labels[j]:
            if len(label_mapping[labels[i]]) < len(label_mapping[labels[j]]):
                i, j = j, i

            connected_mnn[i].add(j)
            connected_mnn[j].add(i)
            j_pos = label_mapping[labels[j]]
            labels[j_pos] = labels[i]
            label_mapping[labels[i]].extend(j_pos)

    return connected_mnn


# Search to find path neighbors
def find_new_nn(
    knn_indices,
    knn_dists,
    knn_indices_pos,
    connected_mnn,
    n_neighbors_max,
    verbose=False,
):
    new_knn_dists = []
    new_knn_indices = []

    for i in range(len(knn_indices)):
        min_distances = []
        min_indices = []

        heap = [(0, i)]
        mapping = {}

        seen = set()
        heapq.heapify(heap)
        while len(min_distances) < n_neighbors_max and len(heap) > 0:
            dist, nn = heapq.heappop(heap)
            if nn == -1:
                continue

            if nn not in seen:
                min_distances.append(dist)
                min_indices.append(nn)
                seen.add(nn)
                neighbor = connected_mnn[nn]

                for nn_nn in neighbor:
                    if nn_nn not in seen:
                        distance = 0
                        if nn_nn in knn_indices_pos[nn]:
                            pos = knn_indices_pos[nn][nn_nn]
                            distance = knn_dists[nn][pos]
                        else:
                            pos = knn_indices_pos[nn_nn][nn]
                            distance = knn_dists[nn_nn][pos]
                        distance += dist
                        if nn_nn not in mapping:
                            mapping[nn_nn] = distance
                            heapq.heappush(heap, (distance, nn_nn))
                        elif mapping[nn_nn] > distance:
                            mapping[nn_nn] = distance
                            heapq.heappush(heap, (distance, nn_nn))

        if len(min_distances) < n_neighbors_max:
            for i in range(n_neighbors_max - len(min_distances)):
                min_indices.append(-1)
                min_distances.append(np.inf)

        new_knn_dists.append(min_distances)
        new_knn_indices.append(min_indices)

        if verbose and i % int(len(knn_dists) / 10) == 0:
            print("\tcompleted ", i, " / ", len(knn_dists), "epochs")
    return new_knn_dists, new_knn_indices


# Calculate the connected mutual nn graph
def mutual_nn_nearest(
    knn_indices,
    knn_dists,
    n_neighbors,
    n_neighbors_max,
    connectivity="min_tree",
    verbose=False,
):
    mutual_nn = {}
    nearest_n = {}

    knn_indices_pos = [None] * len(knn_indices)

    total = 0

    for i, top_vals in enumerate(knn_indices):
        nearest_n[i] = set(top_vals)
        knn_indices_pos[i] = {}
        for pos, nn in enumerate(top_vals):
            knn_indices_pos[i][nn] = pos

    total_mutual_nn = 0
    for i, top_vals in enumerate(knn_indices):
        mutual_nn[i] = set()
        for ind, nn in enumerate(top_vals):
            if nn != -1 and (i in nearest_n[nn] and i != nn):
                mutual_nn[i].add(nn)
                total_mutual_nn += 1

    connected_mnn = create_connected_graph(
        mutual_nn, total_mutual_nn, knn_indices, knn_dists, n_neighbors, connectivity
    )
    new_knn_dists, new_knn_indices = find_new_nn(
        knn_indices, knn_dists, knn_indices_pos, connected_mnn, n_neighbors_max, verbose
    )

    return connected_mnn, mutual_nn, np.array(new_knn_indices), np.array(new_knn_dists)


# update embedding of individual subregion dataset
def update_subregion_embedding(adata):
    adata = adata[
        :,
        ~adata.var_names.str.match(
            r"(^Hla-|^Ig[hjkl]|^Rna|^mt-|^Rp[sl]|^Hb[^(p)]|^Gm)"
        ),
    ]
    adata = adata[:, ~adata.var_names.isin(hk_genes1)]

    # derive shared highly variable genes and run PCA on top
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    scexp.pp.highly_variable_genes(adata, n_top_genes=4000, batch_key=["orig.ident"])
    adata.raw = adata
    sc.pp.regress_out(adata, ['nCount_RNA', 'percent_mito_ribo'])
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')

    embedding = pacmap.PaCMAP(
        n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, apply_pca=False
    )
    adata.obsm["X_pacmap"] = embedding.fit_transform(adata.obsm["X_pca"], init="pca")

    reducer = umap.UMAP(
        densmap=True, n_components=2, random_state=reseed, verbose=False
    )
    reducer.fit(adata.obsm["X_pca"], adata.obs["clusters"].astype(str))
    adata.obsm["X_umap"] = reducer.transform(adata.obsm["X_pca"])

    return adata


######################
##### parameters #####
######################

# Distance Metric to Use
metric = snakemake.params["metric"]
connectivity_model = snakemake.params["connectivity_model"]
PLOTS_DIR = os.path.join("output/figures/resolved_subregions/")

matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica"]
matplotlib.rcParams["figure.max_open_warning"] = 20000

reseed = 42
random.seed(reseed)
np.random.seed(reseed)
random_state = check_random_state(reseed)

verbose = True
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

min_dist = 0.1
a, b = find_ab_params(1.0, min_dist)
n_components = 2
negative_sample_rate = 5
k = int(snakemake.params["k"])
ks = k + 10
signature = snakemake.params["substr_sign"]
cb_fpr = snakemake.params["res"]
max_comm_size = 100


######################
##### Load data ######
######################

## IN:
data_path = snakemake.input["h5ad_pairs"]
housekeeping = f"data/housekeeping_mouse.tsv"

## OUT:
picklefile = snakemake.output["picklefile"]
# de_picklefile = snakemake.output["de_picklefile"]

hk_genes1 = []
with open(housekeeping) as file:
    while (hk_genes := file.readline()):
        hk_genes1.append(hk_genes.rstrip())


PRJNA847050 = sc.read_h5ad(
    f"data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_PRJNA847050-{signature}-astrocytes_datasets_{cb_fpr}.h5ad"
)
PRJNA815819 = sc.read_h5ad(
    f"data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_PRJNA815819-{signature}-astrocytes_datasets_{cb_fpr}.h5ad"
)
PRJNA798401 = sc.read_h5ad(
    f"data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_PRJNA798401-{signature}-astrocytes_datasets_{cb_fpr}.h5ad"
)
PRJNA723345 = sc.read_h5ad(
    f"data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_PRJNA723345-{signature}-astrocytes_datasets_{cb_fpr}.h5ad"
)
PRJNA722418 = sc.read_h5ad(
    f"data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_PRJNA722418-{signature}-astrocytes_datasets_{cb_fpr}.h5ad"
)
PRJNA705596 = sc.read_h5ad(
    f"data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_PRJNA705596-{signature}-astrocytes_datasets_{cb_fpr}.h5ad"
)
PRJNA679294 = sc.read_h5ad(
    f"data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_PRJNA679294-{signature}-astrocytes_datasets_{cb_fpr}.h5ad"
)
PRJNA611624 = sc.read_h5ad(
    f"data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_PRJNA611624-{signature}-astrocytes_datasets_{cb_fpr}.h5ad"
)
PRJNA604055 = sc.read_h5ad(
    f"data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_PRJNA604055-{signature}-astrocytes_datasets_{cb_fpr}.h5ad"
)
PRJNA548532 = sc.read_h5ad(
    f"data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_PRJNA548532-{signature}-astrocytes_datasets_{cb_fpr}.h5ad"
)
PRJNA515063 = sc.read_h5ad(
    f"data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_PRJNA515063-{signature}-astrocytes_datasets_{cb_fpr}.h5ad"
)
PRJNA453138 = sc.read_h5ad(
    f"data/paired_integrations-wo_signature/paired_mtx-PRJNA779749_and_PRJNA453138-{signature}-astrocytes_datasets_{cb_fpr}.h5ad"
)

PRJNA779749_counts = sc.read_h5ad(
    f"PRJNA779749/data/class_cello/PRJNA779749-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad"
)
PRJNA847050_counts = sc.read_h5ad(
    f"PRJNA847050/data/class_cello/PRJNA847050-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad"
)
PRJNA815819_counts = sc.read_h5ad(
    f"PRJNA815819/data/class_cello/PRJNA815819-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad"
)
PRJNA798401_counts = sc.read_h5ad(
    f"PRJNA798401/data/class_cello/PRJNA798401-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad"
)
PRJNA723345_counts = sc.read_h5ad(
    f"PRJNA723345/data/class_cello/PRJNA723345-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad"
)
PRJNA722418_counts = sc.read_h5ad(
    f"PRJNA722418/data/class_cello/PRJNA722418-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad"
)
PRJNA705596_counts = sc.read_h5ad(
    f"PRJNA705596/data/class_cello/PRJNA705596-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad"
)
PRJNA679294_counts = sc.read_h5ad(
    f"PRJNA679294/data/class_cello/PRJNA679294-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad"
)
PRJNA611624_counts = sc.read_h5ad(
    f"PRJNA611624/data/class_cello/PRJNA611624-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad"
)
PRJNA604055_counts = sc.read_h5ad(
    f"PRJNA604055/data/class_cello/PRJNA604055-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad"
)
PRJNA548532_counts = sc.read_h5ad(
    f"PRJNA548532/data/class_cello/PRJNA548532-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad"
)
PRJNA515063_counts = sc.read_h5ad(
    f"PRJNA515063/data/class_cello/PRJNA515063-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad"
)
PRJNA453138_counts = sc.read_h5ad(
    f"PRJNA453138/data/class_cello/PRJNA453138-astrocytes_dataset-{cb_fpr}-initial_selection.h5ad"
)

PRJNA779749_counts = PRJNA779749_counts[PRJNA779749_counts.obs["condit"] == 0]
PRJNA847050_counts = PRJNA847050_counts[PRJNA847050_counts.obs["condit"] == 0]
PRJNA815819_counts = PRJNA815819_counts[PRJNA815819_counts.obs["condit"] == 0]
PRJNA798401_counts = PRJNA798401_counts[PRJNA798401_counts.obs["condit"] == 0]
PRJNA723345_counts = PRJNA723345_counts[PRJNA723345_counts.obs["condit"] == 0]
PRJNA722418_counts = PRJNA722418_counts[PRJNA722418_counts.obs["condit"] == 0]
PRJNA705596_counts = PRJNA705596_counts[PRJNA705596_counts.obs["condit"] == 0]
PRJNA679294_counts = PRJNA679294_counts[PRJNA679294_counts.obs["condit"] == 0]
PRJNA611624_counts = PRJNA611624_counts[PRJNA611624_counts.obs["condit"] == 0]
PRJNA604055_counts = PRJNA604055_counts[PRJNA604055_counts.obs["condit"] == 0]
PRJNA548532_counts = PRJNA548532_counts[PRJNA548532_counts.obs["condit"] == 0]
PRJNA515063_counts = PRJNA515063_counts[PRJNA515063_counts.obs["condit"] == 0]
PRJNA453138_counts = PRJNA453138_counts[PRJNA453138_counts.obs["condit"] == 0]


#######################################
##### derive set of paired graphs #####
#######################################
warnings.filterwarnings("ignore")


# split train and test sets first
X1_PRJNA847050, X2_PRJNA847050, X3_PRJNA847050 = (
    PRJNA847050.obsm["X_pca_harmony"][
        PRJNA847050.obs["train"] & (PRJNA847050.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA847050.obsm["X_pca_harmony"][
        PRJNA847050.obs["test"] & (PRJNA847050.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA847050.obsm["X_pca_harmony"][PRJNA847050.obs["bioproject"] == "PRJNA847050"],
)

# derive knn graph
(
    knn_indices_PRJNA847050,
    knn_dists_PRJNA847050,
    knn_search_index_PRJNA847050,
) = nearest_neighbors(
    X1_PRJNA847050,
    n_neighbors=k,
    metric=metric,
    metric_kwds={},
    angular=False,
    random_state=random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=snakemake.threads,
    verbose=True,
)

# derive mutual nn graph
(
    connected_mnn_PRJNA847050,
    mutual_nn_PRJNA847050,
    new_knn_indices_PRJNA847050,
    new_knn_dists_PRJNA847050,
) = mutual_nn_nearest(
    knn_indices_PRJNA847050,
    knn_dists_PRJNA847050,
    k,
    ks,
    connectivity=connectivity_model,
    verbose=True,
)

# build fuzzy_simplicial_set
G_PRJNA847050, sigmas_PRJNA847050, rhos_PRJNA847050 = fuzzy_simplicial_set(
    X=X1_PRJNA847050,
    n_neighbors=ks,
    metric=metric,
    random_state=random_state,
    knn_indices=new_knn_indices_PRJNA847050,
    knn_dists=new_knn_dists_PRJNA847050,
)


# split train and test sets first
X1_PRJNA815819, X2_PRJNA815819, X3_PRJNA815819 = (
    PRJNA815819.obsm["X_pca_harmony"][
        PRJNA815819.obs["train"] & (PRJNA815819.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA815819.obsm["X_pca_harmony"][
        PRJNA815819.obs["test"] & (PRJNA815819.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA815819.obsm["X_pca_harmony"][PRJNA815819.obs["bioproject"] == "PRJNA815819"],
)

# derive knn graph
(
    knn_indices_PRJNA815819,
    knn_dists_PRJNA815819,
    knn_search_index_PRJNA815819,
) = nearest_neighbors(
    X1_PRJNA815819,
    n_neighbors=k,
    metric=metric,
    metric_kwds={},
    angular=False,
    random_state=random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=snakemake.threads,
    verbose=True,
)

# derive mutual nn graph
(
    connected_mnn_PRJNA815819,
    mutual_nn_PRJNA815819,
    new_knn_indices_PRJNA815819,
    new_knn_dists_PRJNA815819,
) = mutual_nn_nearest(
    knn_indices_PRJNA815819,
    knn_dists_PRJNA815819,
    k,
    ks,
    connectivity=connectivity_model,
    verbose=True,
)

# build fuzzy_simplicial_set
G_PRJNA815819, sigmas_PRJNA815819, rhos_PRJNA815819 = fuzzy_simplicial_set(
    X=X1_PRJNA815819,
    n_neighbors=ks,
    metric=metric,
    random_state=random_state,
    knn_indices=new_knn_indices_PRJNA815819,
    knn_dists=new_knn_dists_PRJNA815819,
)


# split train and test sets first
X1_PRJNA798401, X2_PRJNA798401, X3_PRJNA798401 = (
    PRJNA798401.obsm["X_pca_harmony"][
        PRJNA798401.obs["train"] & (PRJNA798401.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA798401.obsm["X_pca_harmony"][
        PRJNA798401.obs["test"] & (PRJNA798401.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA798401.obsm["X_pca_harmony"][PRJNA798401.obs["bioproject"] == "PRJNA798401"],
)

# derive knn graph
(
    knn_indices_PRJNA798401,
    knn_dists_PRJNA798401,
    knn_search_index_PRJNA798401,
) = nearest_neighbors(
    X1_PRJNA798401,
    n_neighbors=k,
    metric=metric,
    metric_kwds={},
    angular=False,
    random_state=random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=snakemake.threads,
    verbose=True,
)

# derive mutual nn graph
(
    connected_mnn_PRJNA798401,
    mutual_nn_PRJNA798401,
    new_knn_indices_PRJNA798401,
    new_knn_dists_PRJNA798401,
) = mutual_nn_nearest(
    knn_indices_PRJNA798401,
    knn_dists_PRJNA798401,
    k,
    ks,
    connectivity=connectivity_model,
    verbose=True,
)

# build fuzzy_simplicial_set
G_PRJNA798401, sigmas_PRJNA798401, rhos_PRJNA798401 = fuzzy_simplicial_set(
    X=X1_PRJNA798401,
    n_neighbors=ks,
    metric=metric,
    random_state=random_state,
    knn_indices=new_knn_indices_PRJNA798401,
    knn_dists=new_knn_dists_PRJNA798401,
)


# split train and test sets first
X1_PRJNA723345, X2_PRJNA723345, X3_PRJNA723345 = (
    PRJNA723345.obsm["X_pca_harmony"][
        PRJNA723345.obs["train"] & (PRJNA723345.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA723345.obsm["X_pca_harmony"][
        PRJNA723345.obs["test"] & (PRJNA723345.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA723345.obsm["X_pca_harmony"][PRJNA723345.obs["bioproject"] == "PRJNA723345"],
)

# derive knn graph
(
    knn_indices_PRJNA723345,
    knn_dists_PRJNA723345,
    knn_search_index_PRJNA723345,
) = nearest_neighbors(
    X1_PRJNA723345,
    n_neighbors=k,
    metric=metric,
    metric_kwds={},
    angular=False,
    random_state=random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=snakemake.threads,
    verbose=True,
)

# derive mutual nn graph
(
    connected_mnn_PRJNA723345,
    mutual_nn_PRJNA723345,
    new_knn_indices_PRJNA723345,
    new_knn_dists_PRJNA723345,
) = mutual_nn_nearest(
    knn_indices_PRJNA723345,
    knn_dists_PRJNA723345,
    k,
    ks,
    connectivity=connectivity_model,
    verbose=True,
)

# build fuzzy_simplicial_set
G_PRJNA723345, sigmas_PRJNA723345, rhos_PRJNA723345 = fuzzy_simplicial_set(
    X=X1_PRJNA723345,
    n_neighbors=ks,
    metric=metric,
    random_state=random_state,
    knn_indices=new_knn_indices_PRJNA723345,
    knn_dists=new_knn_dists_PRJNA723345,
)


# split train and test sets first
X1_PRJNA722418, X2_PRJNA722418, X3_PRJNA722418 = (
    PRJNA722418.obsm["X_pca_harmony"][
        PRJNA722418.obs["train"] & (PRJNA722418.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA722418.obsm["X_pca_harmony"][
        PRJNA722418.obs["test"] & (PRJNA722418.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA722418.obsm["X_pca_harmony"][PRJNA722418.obs["bioproject"] == "PRJNA722418"],
)

# derive knn graph
(
    knn_indices_PRJNA722418,
    knn_dists_PRJNA722418,
    knn_search_index_PRJNA722418,
) = nearest_neighbors(
    X1_PRJNA722418,
    n_neighbors=k,
    metric=metric,
    metric_kwds={},
    angular=False,
    random_state=random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=snakemake.threads,
    verbose=True,
)

# derive mutual nn graph
(
    connected_mnn_PRJNA722418,
    mutual_nn_PRJNA722418,
    new_knn_indices_PRJNA722418,
    new_knn_dists_PRJNA722418,
) = mutual_nn_nearest(
    knn_indices_PRJNA722418,
    knn_dists_PRJNA722418,
    k,
    ks,
    connectivity=connectivity_model,
    verbose=True,
)

# build fuzzy_simplicial_set
G_PRJNA722418, sigmas_PRJNA722418, rhos_PRJNA722418 = fuzzy_simplicial_set(
    X=X1_PRJNA722418,
    n_neighbors=ks,
    metric=metric,
    random_state=random_state,
    knn_indices=new_knn_indices_PRJNA722418,
    knn_dists=new_knn_dists_PRJNA722418,
)


# split train and test sets first
X1_PRJNA705596, X2_PRJNA705596, X3_PRJNA705596 = (
    PRJNA705596.obsm["X_pca_harmony"][
        PRJNA705596.obs["train"] & (PRJNA705596.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA705596.obsm["X_pca_harmony"][
        PRJNA705596.obs["test"] & (PRJNA705596.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA705596.obsm["X_pca_harmony"][PRJNA705596.obs["bioproject"] == "PRJNA705596"],
)

# derive knn graph
(
    knn_indices_PRJNA705596,
    knn_dists_PRJNA705596,
    knn_search_index_PRJNA705596,
) = nearest_neighbors(
    X1_PRJNA705596,
    n_neighbors=k,
    metric=metric,
    metric_kwds={},
    angular=False,
    random_state=random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=snakemake.threads,
    verbose=True,
)

# derive mutual nn graph
(
    connected_mnn_PRJNA705596,
    mutual_nn_PRJNA705596,
    new_knn_indices_PRJNA705596,
    new_knn_dists_PRJNA705596,
) = mutual_nn_nearest(
    knn_indices_PRJNA705596,
    knn_dists_PRJNA705596,
    k,
    ks,
    connectivity=connectivity_model,
    verbose=True,
)

# build fuzzy_simplicial_set
G_PRJNA705596, sigmas_PRJNA705596, rhos_PRJNA705596 = fuzzy_simplicial_set(
    X=X1_PRJNA705596,
    n_neighbors=ks,
    metric=metric,
    random_state=random_state,
    knn_indices=new_knn_indices_PRJNA705596,
    knn_dists=new_knn_dists_PRJNA705596,
)


# split train and test sets first
X1_PRJNA679294, X2_PRJNA679294, X3_PRJNA679294 = (
    PRJNA679294.obsm["X_pca_harmony"][
        PRJNA679294.obs["train"] & (PRJNA679294.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA679294.obsm["X_pca_harmony"][
        PRJNA679294.obs["test"] & (PRJNA679294.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA679294.obsm["X_pca_harmony"][PRJNA679294.obs["bioproject"] == "PRJNA679294"],
)

# derive knn graph
(
    knn_indices_PRJNA679294,
    knn_dists_PRJNA679294,
    knn_search_index_PRJNA679294,
) = nearest_neighbors(
    X1_PRJNA679294,
    n_neighbors=k,
    metric=metric,
    metric_kwds={},
    angular=False,
    random_state=random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=snakemake.threads,
    verbose=True,
)

# derive mutual nn graph
(
    connected_mnn_PRJNA679294,
    mutual_nn_PRJNA679294,
    new_knn_indices_PRJNA679294,
    new_knn_dists_PRJNA679294,
) = mutual_nn_nearest(
    knn_indices_PRJNA679294,
    knn_dists_PRJNA679294,
    k,
    ks,
    connectivity=connectivity_model,
    verbose=True,
)

# build fuzzy_simplicial_set
G_PRJNA679294, sigmas_PRJNA679294, rhos_PRJNA679294 = fuzzy_simplicial_set(
    X=X1_PRJNA679294,
    n_neighbors=ks,
    metric=metric,
    random_state=random_state,
    knn_indices=new_knn_indices_PRJNA679294,
    knn_dists=new_knn_dists_PRJNA679294,
)


# split train and test sets first
X1_PRJNA611624, X2_PRJNA611624, X3_PRJNA611624 = (
    PRJNA611624.obsm["X_pca_harmony"][
        PRJNA611624.obs["train"] & (PRJNA611624.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA611624.obsm["X_pca_harmony"][
        PRJNA611624.obs["test"] & (PRJNA611624.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA611624.obsm["X_pca_harmony"][PRJNA611624.obs["bioproject"] == "PRJNA611624"],
)

# derive knn graph
(
    knn_indices_PRJNA611624,
    knn_dists_PRJNA611624,
    knn_search_index_PRJNA611624,
) = nearest_neighbors(
    X1_PRJNA611624,
    n_neighbors=k,
    metric=metric,
    metric_kwds={},
    angular=False,
    random_state=random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=snakemake.threads,
    verbose=True,
)

# derive mutual nn graph
(
    connected_mnn_PRJNA611624,
    mutual_nn_PRJNA611624,
    new_knn_indices_PRJNA611624,
    new_knn_dists_PRJNA611624,
) = mutual_nn_nearest(
    knn_indices_PRJNA611624,
    knn_dists_PRJNA611624,
    k,
    ks,
    connectivity=connectivity_model,
    verbose=True,
)

# build fuzzy_simplicial_set
G_PRJNA611624, sigmas_PRJNA611624, rhos_PRJNA611624 = fuzzy_simplicial_set(
    X=X1_PRJNA611624,
    n_neighbors=ks,
    metric=metric,
    random_state=random_state,
    knn_indices=new_knn_indices_PRJNA611624,
    knn_dists=new_knn_dists_PRJNA611624,
)


# split train and test sets first
X1_PRJNA604055, X2_PRJNA604055, X3_PRJNA604055 = (
    PRJNA604055.obsm["X_pca_harmony"][
        PRJNA604055.obs["train"] & (PRJNA604055.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA604055.obsm["X_pca_harmony"][
        PRJNA604055.obs["test"] & (PRJNA604055.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA604055.obsm["X_pca_harmony"][PRJNA604055.obs["bioproject"] == "PRJNA604055"],
)

# derive knn graph
(
    knn_indices_PRJNA604055,
    knn_dists_PRJNA604055,
    knn_search_index_PRJNA604055,
) = nearest_neighbors(
    X1_PRJNA604055,
    n_neighbors=k,
    metric=metric,
    metric_kwds={},
    angular=False,
    random_state=random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=snakemake.threads,
    verbose=True,
)

# derive mutual nn graph
(
    connected_mnn_PRJNA604055,
    mutual_nn_PRJNA604055,
    new_knn_indices_PRJNA604055,
    new_knn_dists_PRJNA604055,
) = mutual_nn_nearest(
    knn_indices_PRJNA604055,
    knn_dists_PRJNA604055,
    k,
    ks,
    connectivity=connectivity_model,
    verbose=True,
)

# build fuzzy_simplicial_set
G_PRJNA604055, sigmas_PRJNA604055, rhos_PRJNA604055 = fuzzy_simplicial_set(
    X=X1_PRJNA604055,
    n_neighbors=ks,
    metric=metric,
    random_state=random_state,
    knn_indices=new_knn_indices_PRJNA604055,
    knn_dists=new_knn_dists_PRJNA604055,
)


# split train and test sets first
X1_PRJNA548532, X2_PRJNA548532, X3_PRJNA548532 = (
    PRJNA548532.obsm["X_pca_harmony"][
        PRJNA548532.obs["train"] & (PRJNA548532.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA548532.obsm["X_pca_harmony"][
        PRJNA548532.obs["test"] & (PRJNA548532.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA548532.obsm["X_pca_harmony"][PRJNA548532.obs["bioproject"] == "PRJNA548532"],
)

# derive knn graph
(
    knn_indices_PRJNA548532,
    knn_dists_PRJNA548532,
    knn_search_index_PRJNA548532,
) = nearest_neighbors(
    X1_PRJNA548532,
    n_neighbors=k,
    metric=metric,
    metric_kwds={},
    angular=False,
    random_state=random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=snakemake.threads,
    verbose=True,
)

# derive mutual nn graph
(
    connected_mnn_PRJNA548532,
    mutual_nn_PRJNA548532,
    new_knn_indices_PRJNA548532,
    new_knn_dists_PRJNA548532,
) = mutual_nn_nearest(
    knn_indices_PRJNA548532,
    knn_dists_PRJNA548532,
    k,
    ks,
    connectivity=connectivity_model,
    verbose=True,
)

# build fuzzy_simplicial_set
G_PRJNA548532, sigmas_PRJNA548532, rhos_PRJNA548532 = fuzzy_simplicial_set(
    X=X1_PRJNA548532,
    n_neighbors=ks,
    metric=metric,
    random_state=random_state,
    knn_indices=new_knn_indices_PRJNA548532,
    knn_dists=new_knn_dists_PRJNA548532,
)


# split train and test sets first
X1_PRJNA515063, X2_PRJNA515063, X3_PRJNA515063 = (
    PRJNA515063.obsm["X_pca_harmony"][
        PRJNA515063.obs["train"] & (PRJNA515063.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA515063.obsm["X_pca_harmony"][
        PRJNA515063.obs["test"] & (PRJNA515063.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA515063.obsm["X_pca_harmony"][PRJNA515063.obs["bioproject"] == "PRJNA515063"],
)

# derive knn graph
(
    knn_indices_PRJNA515063,
    knn_dists_PRJNA515063,
    knn_search_index_PRJNA515063,
) = nearest_neighbors(
    X1_PRJNA515063,
    n_neighbors=k,
    metric=metric,
    metric_kwds={},
    angular=False,
    random_state=random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=snakemake.threads,
    verbose=True,
)

# derive mutual nn graph
(
    connected_mnn_PRJNA515063,
    mutual_nn_PRJNA515063,
    new_knn_indices_PRJNA515063,
    new_knn_dists_PRJNA515063,
) = mutual_nn_nearest(
    knn_indices_PRJNA515063,
    knn_dists_PRJNA515063,
    k,
    ks,
    connectivity=connectivity_model,
    verbose=True,
)

# build fuzzy_simplicial_set
G_PRJNA515063, sigmas_PRJNA515063, rhos_PRJNA515063 = fuzzy_simplicial_set(
    X=X1_PRJNA515063,
    n_neighbors=ks,
    metric=metric,
    random_state=random_state,
    knn_indices=new_knn_indices_PRJNA515063,
    knn_dists=new_knn_dists_PRJNA515063,
)


# split train and test sets first
X1_PRJNA453138, X2_PRJNA453138, X3_PRJNA453138 = (
    PRJNA453138.obsm["X_pca_harmony"][
        PRJNA453138.obs["train"] & (PRJNA453138.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA453138.obsm["X_pca_harmony"][
        PRJNA453138.obs["test"] & (PRJNA453138.obs["bioproject"] == "PRJNA779749")
    ],
    PRJNA453138.obsm["X_pca_harmony"][PRJNA453138.obs["bioproject"] == "PRJNA453138"],
)

# derive knn graph
(
    knn_indices_PRJNA453138,
    knn_dists_PRJNA453138,
    knn_search_index_PRJNA453138,
) = nearest_neighbors(
    X1_PRJNA453138,
    n_neighbors=k,
    metric=metric,
    metric_kwds={},
    angular=False,
    random_state=random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=snakemake.threads,
    verbose=True,
)

# derive mutual nn graph
(
    connected_mnn_PRJNA453138,
    mutual_nn_PRJNA453138,
    new_knn_indices_PRJNA453138,
    new_knn_dists_PRJNA453138,
) = mutual_nn_nearest(
    knn_indices_PRJNA453138,
    knn_dists_PRJNA453138,
    k,
    ks,
    connectivity=connectivity_model,
    verbose=True,
)

# build fuzzy_simplicial_set
G_PRJNA453138, sigmas_PRJNA453138, rhos_PRJNA453138 = fuzzy_simplicial_set(
    X=X1_PRJNA453138,
    n_neighbors=ks,
    metric=metric,
    random_state=random_state,
    knn_indices=new_knn_indices_PRJNA453138,
    knn_dists=new_knn_dists_PRJNA453138,
)


####################################
##### derive multiplex graph #####
##################################
wc = 1 - PRJNA779749_counts.shape[0]/PRJNA798401_counts.shape[0]

PRJNA847050_w = (1 - PRJNA847050_counts.shape[0]/PRJNA798401_counts.shape[0]) * wc
PRJNA815819_w = (1 - PRJNA815819_counts.shape[0]/PRJNA798401_counts.shape[0]) * wc
PRJNA798401_w = PRJNA779749_counts.shape[0]/PRJNA798401_counts.shape[0]
PRJNA723345_w = (1 - PRJNA723345_counts.shape[0]/PRJNA798401_counts.shape[0]) * wc
PRJNA722418_w = (1 - PRJNA722418_counts.shape[0]/PRJNA798401_counts.shape[0]) * wc
PRJNA705596_w = (1 - PRJNA705596_counts.shape[0]/PRJNA798401_counts.shape[0]) * wc
PRJNA679294_w = (1 - PRJNA679294_counts.shape[0]/PRJNA798401_counts.shape[0]) * wc
PRJNA611624_w = (1 - PRJNA611624_counts.shape[0]/PRJNA798401_counts.shape[0]) * wc
PRJNA604055_w = (1 - PRJNA604055_counts.shape[0]/PRJNA798401_counts.shape[0]) * wc
PRJNA548532_w = (1 - PRJNA548532_counts.shape[0]/PRJNA798401_counts.shape[0]) * wc
PRJNA515063_w = (1 - PRJNA515063_counts.shape[0]/PRJNA798401_counts.shape[0]) * wc
PRJNA453138_w = (1 - PRJNA453138_counts.shape[0]/PRJNA798401_counts.shape[0]) * wc


part_PRJNA847050 = la.ModularityVertexPartition(ig.Graph.from_networkx(nx.from_scipy_sparse_array(G_PRJNA847050)), initial_membership=PRJNA847050[PRJNA847050.obs['train'] & (PRJNA847050.obs["bioproject"] == "PRJNA779749")].obs['leiden'].astype(int).tolist(), weights='weight')
part_PRJNA815819 = la.ModularityVertexPartition(ig.Graph.from_networkx(nx.from_scipy_sparse_array(G_PRJNA815819)), initial_membership=PRJNA815819[PRJNA815819.obs['train'] & (PRJNA815819.obs["bioproject"] == "PRJNA779749")].obs['leiden'].astype(int).tolist(), weights='weight')
part_PRJNA798401 = la.ModularityVertexPartition(ig.Graph.from_networkx(nx.from_scipy_sparse_array(G_PRJNA798401)), initial_membership=PRJNA798401[PRJNA798401.obs['train'] & (PRJNA798401.obs["bioproject"] == "PRJNA779749")].obs['leiden'].astype(int).tolist(), weights='weight')
part_PRJNA723345 = la.ModularityVertexPartition(ig.Graph.from_networkx(nx.from_scipy_sparse_array(G_PRJNA723345)), initial_membership=PRJNA723345[PRJNA723345.obs['train'] & (PRJNA723345.obs["bioproject"] == "PRJNA779749")].obs['leiden'].astype(int).tolist(), weights='weight')
part_PRJNA722418 = la.ModularityVertexPartition(ig.Graph.from_networkx(nx.from_scipy_sparse_array(G_PRJNA722418)), initial_membership=PRJNA722418[PRJNA722418.obs['train'] & (PRJNA722418.obs["bioproject"] == "PRJNA779749")].obs['leiden'].astype(int).tolist(), weights='weight')
part_PRJNA705596 = la.ModularityVertexPartition(ig.Graph.from_networkx(nx.from_scipy_sparse_array(G_PRJNA705596)), initial_membership=PRJNA705596[PRJNA705596.obs['train'] & (PRJNA705596.obs["bioproject"] == "PRJNA779749")].obs['leiden'].astype(int).tolist(), weights='weight')
part_PRJNA679294 = la.ModularityVertexPartition(ig.Graph.from_networkx(nx.from_scipy_sparse_array(G_PRJNA679294)), initial_membership=PRJNA679294[PRJNA679294.obs['train'] & (PRJNA679294.obs["bioproject"] == "PRJNA779749")].obs['leiden'].astype(int).tolist(), weights='weight')
part_PRJNA611624 = la.ModularityVertexPartition(ig.Graph.from_networkx(nx.from_scipy_sparse_array(G_PRJNA611624)), initial_membership=PRJNA611624[PRJNA611624.obs['train'] & (PRJNA611624.obs["bioproject"] == "PRJNA779749")].obs['leiden'].astype(int).tolist(), weights='weight')
part_PRJNA604055 = la.ModularityVertexPartition(ig.Graph.from_networkx(nx.from_scipy_sparse_array(G_PRJNA604055)), initial_membership=PRJNA604055[PRJNA604055.obs['train'] & (PRJNA604055.obs["bioproject"] == "PRJNA779749")].obs['leiden'].astype(int).tolist(), weights='weight')
part_PRJNA548532 = la.ModularityVertexPartition(ig.Graph.from_networkx(nx.from_scipy_sparse_array(G_PRJNA548532)), initial_membership=PRJNA548532[PRJNA548532.obs['train'] & (PRJNA548532.obs["bioproject"] == "PRJNA779749")].obs['leiden'].astype(int).tolist(), weights='weight')
part_PRJNA515063 = la.ModularityVertexPartition(ig.Graph.from_networkx(nx.from_scipy_sparse_array(G_PRJNA515063)), initial_membership=PRJNA515063[PRJNA515063.obs['train'] & (PRJNA515063.obs["bioproject"] == "PRJNA779749")].obs['leiden'].astype(int).tolist(), weights='weight')
part_PRJNA453138 = la.ModularityVertexPartition(ig.Graph.from_networkx(nx.from_scipy_sparse_array(G_PRJNA453138)), initial_membership=PRJNA453138[PRJNA453138.obs['train'] & (PRJNA453138.obs["bioproject"] == "PRJNA779749")].obs['leiden'].astype(int).tolist(), weights='weight')

optimiser = la.Optimiser()
optimiser.set_rng_seed(reseed)
optimiser.max_comm_size = max_comm_size

diff = optimiser.optimise_partition_multiplex(
  partitions = [
        part_PRJNA847050,
        part_PRJNA815819,
        part_PRJNA798401,
        part_PRJNA723345,
        part_PRJNA722418,
        part_PRJNA705596,
        part_PRJNA679294,
        part_PRJNA611624,
        part_PRJNA604055,
        part_PRJNA548532,
        part_PRJNA515063,
        part_PRJNA453138
  ],
  layer_weights=[
        PRJNA847050_w,
        PRJNA815819_w,
        PRJNA798401_w,
        PRJNA723345_w,
        PRJNA722418_w,
        PRJNA705596_w,
        PRJNA679294_w,
        PRJNA611624_w,
        PRJNA604055_w,
        PRJNA548532_w,
        PRJNA515063_w,
        PRJNA453138_w
    ],
    n_iterations=-1
)

membership = part_PRJNA847050.membership

##########################################################
##### train Density preserving supervised UMAP model #####
##########################################################
reducer_by_PRJNA847050 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_by_PRJNA847050.fit(X1_PRJNA847050, membership)

X1_PRJNA847050_densmap_supervised = reducer_by_PRJNA847050.transform(X1_PRJNA847050)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    X1_PRJNA847050_densmap_supervised[:, 0],
    X1_PRJNA847050_densmap_supervised[:, 1],
    c=membership,
    cmap="nipy_spectral",
    edgecolor="k",
    label=membership,
)
plt.colorbar(boundaries=np.arange(64) - 0.5).set_ticks(np.arange(63))
plt.savefig(
    os.path.join(
        PLOTS_DIR,
        f"astrocytes_X1_PRJNA847050-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-densmap_supervised.pdf",
    )
)


# Fit linear separator using X_train
svm_PRJNA847050 = SVC(kernel="linear", C=100)
svm_PRJNA847050.fit(X1_PRJNA847050, membership)
joblib.dump(
    svm_PRJNA847050,
    f"models/svm/PRJNA847050-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pickle",
)

# Generate labels for X_test
predictions_X2_PRJNA847050 = svm_PRJNA847050.predict(X2_PRJNA847050)
predictions_X3_PRJNA847050 = svm_PRJNA847050.predict(X3_PRJNA847050)


PRJNA847050.obs["clusters"] = PRJNA847050.obs["condit"]
PRJNA847050.obs["clusters"][
    PRJNA847050.obs["train"] & (PRJNA847050.obs["bioproject"] == "PRJNA779749")
] = membership
PRJNA847050.obs["clusters"][
    PRJNA847050.obs["test"] & (PRJNA847050.obs["bioproject"] == "PRJNA779749")
] = predictions_X2_PRJNA847050
PRJNA847050.obs["clusters"][
    PRJNA847050.obs["bioproject"] == "PRJNA847050"
] = predictions_X3_PRJNA847050
PRJNA847050.obs["clusters"] = pd.Categorical(
        values=PRJNA847050.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA847050.obs["clusters"].astype(str)))),
    )
PRJNA847050_counts.obs["clusters"] = predictions_X3_PRJNA847050
PRJNA847050_counts.obs["clusters"] = pd.Categorical(
        values=PRJNA847050_counts.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA847050_counts.obs["clusters"].astype(str)))),
    )


reducer_PRJNA847050 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_PRJNA847050.fit(PRJNA847050.obsm["X_pca_harmony"], PRJNA847050.obs["clusters"].astype(str))
PRJNA847050.obsm["X_umap"] = reducer_PRJNA847050.transform(
    PRJNA847050.obsm["X_pca_harmony"]
)

save_PRJNA847050 = f"-supervised-PRJNA847050-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.embedding(
    PRJNA847050,
    basis="X_pacmap",
    color=["project", "clusters"],
    title=f"PaCMAP: paired integration, PRJNA847050 (amb.FPR={cb_fpr}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save=save_PRJNA847050,
)
print(PRJNA847050.obs['clusters'].value_counts().sort_values())


sc.pl.umap(
    PRJNA847050,
    color=["project"],
    title=f"UMAP: supervised integration, PRJNA847050 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-project" + save_PRJNA847050,
)

sc.tl.embedding_density(PRJNA847050, basis='umap', groupby='project')
sc.pl.embedding_density(
    PRJNA847050,
    basis='umap',
    key='umap_density_project',
    save=save_PRJNA847050)


sc.pl.umap(
    PRJNA847050,
    color=["clusters"],
    title=f"UMAP: supervised integration, PRJNA847050 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-clusters" + save_PRJNA847050,
)

sc.tl.embedding_density(PRJNA847050, basis='umap', groupby='clusters')
sc.pl.embedding_density(
    PRJNA847050,
    basis='umap',
    key='umap_density_clusters',
    save=save_PRJNA847050)

PRJNA847050.write(
    f"data/resolved_subregions/paired_mtx-PRJNA779749_and_PRJNA847050-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)

PRJNA847050_counts = update_subregion_embedding(PRJNA847050_counts)
PRJNA847050_counts.write(
    f"data/resolved_subregions/PRJNA847050-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA847050_counts.write_loom(
    f"data/resolved_subregions/PRJNA847050-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.loom"
)
pd.DataFrame(
    PRJNA847050_counts.obsm["X_umap"], index=PRJNA847050_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA847050-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-umap.tsv",
    sep="\t",
    header=True,
)
pd.DataFrame(
    PRJNA847050_counts.obsm["X_pacmap"], index=PRJNA847050_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA847050-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-pacmap.tsv",
    sep="\t",
    header=True,
)


sc.pl.umap(
    PRJNA847050, color=[g for g in npep if g in PRJNA847050.var_names], frameon=False, show=False, save="-npep" + save_PRJNA847050
)


sc.pl.umap(
    PRJNA847050,
    color=[g for g in npr if g in PRJNA847050.var_names],
    add_outline=True,
    frameon=False,
    show=False,
    save="-npr" + save_PRJNA847050,
)

sc.pl.umap(
    PRJNA847050,
    color=[g for g in nmr if g in PRJNA847050.var_names],
    add_outline=True,
    frameon=False,
    show=False,
    save="-nmr" + save_PRJNA847050,
)


sc.pl.umap(
    PRJNA847050,
    color=[g for g in genes_embed if g in PRJNA847050.var_names],
    frameon=False,
    show=False,
    save="-adgen" + save_PRJNA847050,
)

PRJNA847050.obs["learning"] = pd.Categorical(
    ["train" if x else "test" for x in PRJNA847050.obs["train"]]
)
outlier_scores = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=ks, contamination=0.1
).fit_predict(PRJNA847050.obsm["X_umap"])


PRJNA847050_outlying = PRJNA847050[outlier_scores == -1]
PRJNA847050_outlying.shape


PRJNA847050_outlying


sc.pl.umap(
    PRJNA847050_outlying,
    color=[g for g in genes_embed if g in PRJNA847050.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save=False,
)


sc.pl.umap(
    PRJNA847050_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA847050 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_outlying" + save_PRJNA847050,
)


sc.pl.umap(
    PRJNA847050_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA847050 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_outlying" + save_PRJNA847050,
)


PRJNA847050_wo_outlying = PRJNA847050[outlier_scores != -1]
PRJNA847050_wo_outlying.shape


sc.pl.umap(
    PRJNA847050_wo_outlying,
    color=[g for g in genes_embed if g in PRJNA847050.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgene_wo_outlying" + save_PRJNA847050,
)


sc.pl.umap(
    PRJNA847050_wo_outlying,
    color=[g for g in npep if g in PRJNA847050.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep_wo_outlying" + save_PRJNA847050,
)


sc.pl.umap(
    PRJNA847050_wo_outlying,
    color=[g for g in npr if g in PRJNA847050.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr_wo_outlying" + save_PRJNA847050,
)


sc.pl.umap(
    PRJNA847050_wo_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA847050 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_wo_outlying" + save_PRJNA847050,
)


sc.pl.umap(
    PRJNA847050_wo_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA847050 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_wo_outlying" + save_PRJNA847050,
)


##########################################################
##### train Density preserving supervised UMAP model #####
##########################################################
reducer_by_PRJNA815819 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_by_PRJNA815819.fit(X1_PRJNA815819, membership)

X1_PRJNA815819_densmap_supervised = reducer_by_PRJNA815819.transform(X1_PRJNA815819)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    X1_PRJNA815819_densmap_supervised[:, 0],
    X1_PRJNA815819_densmap_supervised[:, 1],
    c=membership,
    cmap="nipy_spectral",
    edgecolor="k",
    label=membership,
)
plt.colorbar(boundaries=np.arange(64) - 0.5).set_ticks(np.arange(63))
plt.savefig(
    os.path.join(
        PLOTS_DIR,
        f"astrocytes_X1_PRJNA815819-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-densmap_supervised.pdf",
    )
)


# Fit linear separator using X_train
svm_PRJNA815819 = SVC(kernel="linear", C=100)
svm_PRJNA815819.fit(X1_PRJNA815819, membership)
joblib.dump(
    svm_PRJNA815819,
    f"models/svm/PRJNA815819-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pickle",
)

# Generate labels for X_test
predictions_X2_PRJNA815819 = svm_PRJNA815819.predict(X2_PRJNA815819)
predictions_X3_PRJNA815819 = svm_PRJNA815819.predict(X3_PRJNA815819)


PRJNA815819.obs["clusters"] = PRJNA815819.obs["condit"]
PRJNA815819.obs["clusters"][
    PRJNA815819.obs["train"] & (PRJNA815819.obs["bioproject"] == "PRJNA779749")
] = membership
PRJNA815819.obs["clusters"][
    PRJNA815819.obs["test"] & (PRJNA815819.obs["bioproject"] == "PRJNA779749")
] = predictions_X2_PRJNA815819
PRJNA815819.obs["clusters"][
    PRJNA815819.obs["bioproject"] == "PRJNA815819"
] = predictions_X3_PRJNA815819
PRJNA815819.obs["clusters"] = pd.Categorical(
        values=PRJNA815819.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA815819.obs["clusters"].astype(str)))),
    )
PRJNA815819_counts.obs["clusters"] = predictions_X3_PRJNA815819
PRJNA815819_counts.obs["clusters"] = pd.Categorical(
        values=PRJNA815819_counts.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA815819_counts.obs["clusters"].astype(str)))),
    )


reducer_PRJNA815819 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_PRJNA815819.fit(PRJNA815819.obsm["X_pca_harmony"], PRJNA815819.obs["clusters"].astype(str))
PRJNA815819.obsm["X_umap"] = reducer_PRJNA815819.transform(
    PRJNA815819.obsm["X_pca_harmony"]
)

save_PRJNA815819 = f"-supervised-PRJNA815819-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.embedding(
    PRJNA815819,
    basis="X_pacmap",
    color=["project", "clusters"],
    title=f"PaCMAP: paired integration, PRJNA815819 (amb.FPR={cb_fpr}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save=save_PRJNA815819,
)
print(PRJNA815819.obs['clusters'].value_counts().sort_values())

sc.pl.umap(
    PRJNA815819,
    color=["project"],
    title=f"UMAP: supervised integration, PRJNA815819 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-project" + save_PRJNA815819,
)

sc.tl.embedding_density(PRJNA815819, basis='umap', groupby='project')
sc.pl.embedding_density(
    PRJNA815819,
    basis='umap',
    key='umap_density_project',
    save=save_PRJNA815819)


sc.pl.umap(
    PRJNA815819,
    color=["clusters"],
    title=f"UMAP: supervised integration, PRJNA815819 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-clusters" + save_PRJNA815819,
)

sc.tl.embedding_density(PRJNA815819, basis='umap', groupby='clusters')
sc.pl.embedding_density(
    PRJNA815819,
    basis='umap',
    key='umap_density_clusters',
    save=save_PRJNA815819)


PRJNA815819.write(
    f"data/resolved_subregions/paired_mtx-PRJNA779749_and_PRJNA815819-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)

PRJNA815819_counts = update_subregion_embedding(PRJNA815819_counts)
PRJNA815819_counts.write(
    f"data/resolved_subregions/PRJNA815819-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA815819_counts.write_loom(
    f"data/resolved_subregions/PRJNA815819-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.loom"
)
pd.DataFrame(
    PRJNA815819_counts.obsm["X_umap"], index=PRJNA815819_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA815819-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-umap.tsv",
    sep="\t",
    header=True,
)
pd.DataFrame(
    PRJNA815819_counts.obsm["X_pacmap"], index=PRJNA815819_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA815819-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-pacmap.tsv",
    sep="\t",
    header=True,
)


sc.pl.umap(
    PRJNA815819,
    color=[g for g in npep if g in PRJNA815819.var_names],
    frameon=False,
    show=False,
    save="-npep" + save_PRJNA815819,
)


sc.pl.umap(
    PRJNA815819,
    color=[g for g in npr if g in PRJNA815819.var_names],
    frameon=False,
    show=False,
    save="-npr" + save_PRJNA815819,
)

sc.pl.umap(
    PRJNA815819,
    color=[g for g in nmr if g in PRJNA815819.var_names],
    frameon=False,
    show=False,
    save="-nmr" + save_PRJNA815819,
)


sc.pl.umap(
    PRJNA815819,
    color=[g for g in genes_embed if g in PRJNA815819.var_names],
    add_outline=True,
    frameon=False,
    show=False,
    save="-adgen" + save_PRJNA815819,
)


PRJNA815819.obs["learning"] = pd.Categorical(
    ["train" if x else "test" for x in PRJNA815819.obs["train"]]
)
outlier_scores = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=ks, contamination=0.1
).fit_predict(PRJNA815819.obsm["X_umap"])


PRJNA815819_outlying = PRJNA815819[outlier_scores == -1]
PRJNA815819_outlying.shape


PRJNA815819_outlying


sc.pl.umap(
    PRJNA815819_outlying,
    color=[g for g in genes_embed if g in PRJNA815819.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save=False,
)


sc.pl.umap(
    PRJNA815819_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA815819 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_outlying" + save_PRJNA815819,
)


sc.pl.umap(
    PRJNA815819_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA815819 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_outlying" + save_PRJNA815819,
)


PRJNA815819_wo_outlying = PRJNA815819[outlier_scores != -1]
PRJNA815819_wo_outlying.shape


sc.pl.umap(
    PRJNA815819_wo_outlying,
    color=[g for g in genes_embed if g in PRJNA815819.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgene_wo_outlying" + save_PRJNA815819,
)


sc.pl.umap(
    PRJNA815819_wo_outlying,
    color=[g for g in npep if g in PRJNA815819.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep_wo_outlying" + save_PRJNA815819,
)


sc.pl.umap(
    PRJNA815819_wo_outlying,
    color=[g for g in npr if g in PRJNA815819.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr_wo_outlying" + save_PRJNA815819,
)


sc.pl.umap(
    PRJNA815819_wo_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA815819 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_wo_outlying" + save_PRJNA815819,
)


sc.pl.umap(
    PRJNA815819_wo_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA815819 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_wo_outlying" + save_PRJNA815819,
)


##########################################################
##### train Density preserving supervised UMAP model #####
##########################################################
reducer_by_PRJNA798401 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_by_PRJNA798401.fit(X1_PRJNA798401, membership)

X1_PRJNA798401_densmap_supervised = reducer_by_PRJNA798401.transform(X1_PRJNA798401)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    X1_PRJNA798401_densmap_supervised[:, 0],
    X1_PRJNA798401_densmap_supervised[:, 1],
    c=membership,
    cmap="nipy_spectral",
    edgecolor="k",
    label=membership,
)
plt.colorbar(boundaries=np.arange(64) - 0.5).set_ticks(np.arange(63))
plt.savefig(
    os.path.join(
        PLOTS_DIR,
        f"astrocytes_X1_PRJNA798401-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-densmap_supervised.pdf",
    )
)


# Fit linear separator using X_train
svm_PRJNA798401 = SVC(kernel="linear", C=100)
svm_PRJNA798401.fit(X1_PRJNA798401, membership)
joblib.dump(
    svm_PRJNA798401,
    f"models/svm/PRJNA798401-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pickle",
)

# Generate labels for X_test
predictions_X2_PRJNA798401 = svm_PRJNA798401.predict(X2_PRJNA798401)
predictions_X3_PRJNA798401 = svm_PRJNA798401.predict(X3_PRJNA798401)


PRJNA798401.obs["clusters"] = PRJNA798401.obs["condit"]
PRJNA798401.obs["clusters"][
    PRJNA798401.obs["train"] & (PRJNA798401.obs["bioproject"] == "PRJNA779749")
] = membership
PRJNA798401.obs["clusters"][
    PRJNA798401.obs["test"] & (PRJNA798401.obs["bioproject"] == "PRJNA779749")
] = predictions_X2_PRJNA798401
PRJNA798401.obs["clusters"][
    PRJNA798401.obs["bioproject"] == "PRJNA798401"
] = predictions_X3_PRJNA798401
PRJNA798401.obs["clusters"] = pd.Categorical(
        values=PRJNA798401.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA798401.obs["clusters"].astype(str)))),
    )
PRJNA798401_counts.obs["clusters"] = predictions_X3_PRJNA798401
PRJNA798401_counts.obs["clusters"] = pd.Categorical(
        values=PRJNA798401_counts.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA798401_counts.obs["clusters"].astype(str)))),
    )


reducer_PRJNA798401 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_PRJNA798401.fit(PRJNA798401.obsm["X_pca_harmony"], PRJNA798401.obs["clusters"].astype(str))
PRJNA798401.obsm["X_umap"] = reducer_PRJNA798401.transform(
    PRJNA798401.obsm["X_pca_harmony"]
)

save_PRJNA798401 = f"-supervised-PRJNA798401-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.embedding(
    PRJNA798401,
    basis="X_pacmap",
    color=["project", "clusters"],
    title=f"PaCMAP: paired integration, PRJNA798401 (amb.FPR={cb_fpr}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save=save_PRJNA798401,
)
print(PRJNA798401.obs['clusters'].value_counts().sort_values())


sc.pl.umap(
    PRJNA798401,
    color=["project"],
    title=f"UMAP: supervised integration, PRJNA798401 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-project" + save_PRJNA798401,
)

sc.tl.embedding_density(PRJNA798401, basis='umap', groupby='project')
sc.pl.embedding_density(
    PRJNA798401,
    basis='umap',
    key='umap_density_project',
    save=save_PRJNA798401)


sc.pl.umap(
    PRJNA798401,
    color=["clusters"],
    title=f"UMAP: supervised integration, PRJNA798401 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-clusters" + save_PRJNA798401,
)

sc.tl.embedding_density(PRJNA798401, basis='umap', groupby='clusters')
sc.pl.embedding_density(
    PRJNA798401,
    basis='umap',
    key='umap_density_clusters',
    save=save_PRJNA798401)


PRJNA798401.write(
    f"data/resolved_subregions/paired_mtx-PRJNA779749_and_PRJNA798401-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA798401_counts = update_subregion_embedding(PRJNA798401_counts)
PRJNA798401_counts.write(
    f"data/resolved_subregions/PRJNA798401-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA798401_counts.write_loom(
    f"data/resolved_subregions/PRJNA798401-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.loom"
)
pd.DataFrame(
    PRJNA798401_counts.obsm["X_umap"], index=PRJNA798401_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA798401-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-umap.tsv",
    sep="\t",
    header=True,
)
pd.DataFrame(
    PRJNA798401_counts.obsm["X_pacmap"], index=PRJNA798401_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA798401-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-pacmap.tsv",
    sep="\t",
    header=True,
)


sc.pl.umap(
    PRJNA798401,
    color=[g for g in npep if g in PRJNA798401.var_names],
    frameon=False,
    show=False,
    save="-npep" + save_PRJNA798401,
)


sc.pl.umap(
    PRJNA798401,
    color=[g for g in npr if g in PRJNA798401.var_names],
    frameon=False,
    show=False,
    save="-npr" + save_PRJNA798401,
)

sc.pl.umap(
    PRJNA798401,
    color=[g for g in nmr if g in PRJNA798401.var_names],
    frameon=False,
    show=False,
    save="-nmr" + save_PRJNA798401,
)


sc.pl.umap(
    PRJNA798401,
    color=[g for g in genes_embed if g in PRJNA798401.var_names],
    frameon=False,
    show=False,
    save="-adgen" + save_PRJNA798401,
)


PRJNA798401.obs["learning"] = pd.Categorical(
    ["train" if x else "test" for x in PRJNA798401.obs["train"]]
)
outlier_scores = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=ks, contamination=0.1
).fit_predict(PRJNA798401.obsm["X_umap"])


PRJNA798401_outlying = PRJNA798401[outlier_scores == -1]
PRJNA798401_outlying.shape


PRJNA798401_outlying


sc.pl.umap(
    PRJNA798401_outlying,
    color=[g for g in genes_embed if g in PRJNA798401.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save=False,
)


sc.pl.umap(
    PRJNA798401_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA798401 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_outlying" + save_PRJNA798401,
)


sc.pl.umap(
    PRJNA798401_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA798401 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_outlying" + save_PRJNA798401,
)


PRJNA798401_wo_outlying = PRJNA798401[outlier_scores != -1]
PRJNA798401_wo_outlying.shape


sc.pl.umap(
    PRJNA798401_wo_outlying,
    color=[g for g in genes_embed if g in PRJNA798401.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgene_wo_outlying" + save_PRJNA798401,
)


sc.pl.umap(
    PRJNA798401_wo_outlying,
    color=[g for g in npep if g in PRJNA798401.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep_wo_outlying" + save_PRJNA798401,
)


sc.pl.umap(
    PRJNA798401_wo_outlying,
    color=[g for g in npr if g in PRJNA798401.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr_wo_outlying" + save_PRJNA798401,
)


sc.pl.umap(
    PRJNA798401_wo_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA798401 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_wo_outlying" + save_PRJNA798401,
)


sc.pl.umap(
    PRJNA798401_wo_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA798401 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_wo_outlying" + save_PRJNA798401,
)


##########################################################
##### train Density preserving supervised UMAP model #####
##########################################################
reducer_by_PRJNA723345 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_by_PRJNA723345.fit(X1_PRJNA723345, membership)

X1_PRJNA723345_densmap_supervised = reducer_by_PRJNA723345.transform(X1_PRJNA723345)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    X1_PRJNA723345_densmap_supervised[:, 0],
    X1_PRJNA723345_densmap_supervised[:, 1],
    c=membership,
    cmap="nipy_spectral",
    edgecolor="k",
    label=membership,
)
plt.colorbar(boundaries=np.arange(64) - 0.5).set_ticks(np.arange(63))
plt.savefig(
    os.path.join(
        PLOTS_DIR,
        f"astrocytes_X1_PRJNA723345-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-densmap_supervised.pdf",
    )
)


# Fit linear separator using X_train
svm_PRJNA723345 = SVC(kernel="linear", C=100)
svm_PRJNA723345.fit(X1_PRJNA723345, membership)
joblib.dump(
    svm_PRJNA723345,
    f"models/svm/PRJNA723345-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pickle",
)

# Generate labels for X_test
predictions_X2_PRJNA723345 = svm_PRJNA723345.predict(X2_PRJNA723345)
predictions_X3_PRJNA723345 = svm_PRJNA723345.predict(X3_PRJNA723345)


PRJNA723345.obs["clusters"] = PRJNA723345.obs["condit"]
PRJNA723345.obs["clusters"][
    PRJNA723345.obs["train"] & (PRJNA723345.obs["bioproject"] == "PRJNA779749")
] = membership
PRJNA723345.obs["clusters"][
    PRJNA723345.obs["test"] & (PRJNA723345.obs["bioproject"] == "PRJNA779749")
] = predictions_X2_PRJNA723345
PRJNA723345.obs["clusters"][
    PRJNA723345.obs["bioproject"] == "PRJNA723345"
] = predictions_X3_PRJNA723345
PRJNA723345.obs["clusters"] = pd.Categorical(
        values=PRJNA723345.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA723345.obs["clusters"].astype(str)))),
    )
PRJNA723345_counts.obs["clusters"] = predictions_X3_PRJNA723345
PRJNA723345_counts.obs["clusters"] = pd.Categorical(
        values=PRJNA723345_counts.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA723345_counts.obs["clusters"].astype(str)))),
    )


reducer_PRJNA723345 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_PRJNA723345.fit(PRJNA723345.obsm["X_pca_harmony"], PRJNA723345.obs["clusters"].astype(str))
PRJNA723345.obsm["X_umap"] = reducer_PRJNA723345.transform(
    PRJNA723345.obsm["X_pca_harmony"]
)

save_PRJNA723345 = f"-supervised-PRJNA723345-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.embedding(
    PRJNA723345,
    basis="X_pacmap",
    color=["project", "clusters"],
    title=f"PaCMAP: paired integration, PRJNA723345 (amb.FPR={cb_fpr}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save=save_PRJNA723345,
)
print(PRJNA723345.obs['clusters'].value_counts().sort_values())


sc.pl.umap(
    PRJNA723345,
    color=["project"],
    title=f"UMAP: supervised integration, PRJNA723345 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-project" + save_PRJNA723345,
)

sc.tl.embedding_density(PRJNA723345, basis='umap', groupby='project')
sc.pl.embedding_density(
    PRJNA723345,
    basis='umap',
    key='umap_density_project',
    save=save_PRJNA723345)


sc.pl.umap(
    PRJNA723345,
    color=["clusters"],
    title=f"UMAP: supervised integration, PRJNA723345 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-clusters" + save_PRJNA723345,
)

sc.tl.embedding_density(PRJNA723345, basis='umap', groupby='clusters')
sc.pl.embedding_density(
    PRJNA723345,
    basis='umap',
    key='umap_density_clusters',
    save=save_PRJNA723345)


PRJNA723345.write(
    f"data/resolved_subregions/paired_mtx-PRJNA779749_and_PRJNA723345-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)

PRJNA723345_counts = update_subregion_embedding(PRJNA723345_counts)
PRJNA723345_counts.write(
    f"data/resolved_subregions/PRJNA723345-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA723345_counts.write_loom(
    f"data/resolved_subregions/PRJNA723345-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.loom"
)
pd.DataFrame(
    PRJNA723345_counts.obsm["X_umap"], index=PRJNA723345_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA723345-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-umap.tsv",
    sep="\t",
    header=True,
)
pd.DataFrame(
    PRJNA723345_counts.obsm["X_pacmap"], index=PRJNA723345_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA723345-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-pacmap.tsv",
    sep="\t",
    header=True,
)


sc.pl.umap(
    PRJNA723345,
    color=[g for g in npep if g in PRJNA723345.var_names],
    frameon=False,
    show=False,
    save="-npep" + save_PRJNA723345,
)


sc.pl.umap(
    PRJNA723345,
    color=[g for g in npr if g in PRJNA723345.var_names],
    frameon=False,
    show=False,
    save="-npr" + save_PRJNA723345,
)

sc.pl.umap(
    PRJNA723345,
    color=[g for g in nmr if g in PRJNA723345.var_names],
    frameon=False,
    show=False,
    save="-nmr" + save_PRJNA723345,
)


sc.pl.umap(
    PRJNA723345,
    color=[g for g in genes_embed if g in PRJNA723345.var_names],
    frameon=False,
    show=False,
    save="-adgen" + save_PRJNA723345,
)


PRJNA723345.obs["learning"] = pd.Categorical(
    ["train" if x else "test" for x in PRJNA723345.obs["train"]]
)
outlier_scores = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=ks, contamination=0.1
).fit_predict(PRJNA723345.obsm["X_umap"])


PRJNA723345_outlying = PRJNA723345[outlier_scores == -1]
PRJNA723345_outlying.shape


PRJNA723345_outlying


sc.pl.umap(
    PRJNA723345_outlying,
    color=[g for g in genes_embed if g in PRJNA723345.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save=False,
)


sc.pl.umap(
    PRJNA723345_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA723345 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_outlying" + save_PRJNA723345,
)


sc.pl.umap(
    PRJNA723345_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA723345 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_outlying" + save_PRJNA723345,
)


PRJNA723345_wo_outlying = PRJNA723345[outlier_scores != -1]
PRJNA723345_wo_outlying.shape


sc.pl.umap(
    PRJNA723345_wo_outlying,
    color=[g for g in genes_embed if g in PRJNA723345.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgene_wo_outlying" + save_PRJNA723345,
)


sc.pl.umap(
    PRJNA723345_wo_outlying,
    color=[g for g in npep if g in PRJNA723345.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep_wo_outlying" + save_PRJNA723345,
)


sc.pl.umap(
    PRJNA723345_wo_outlying,
    color=[g for g in npr if g in PRJNA723345.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr_wo_outlying" + save_PRJNA723345,
)


sc.pl.umap(
    PRJNA723345_wo_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA723345 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_wo_outlying" + save_PRJNA723345,
)


sc.pl.umap(
    PRJNA723345_wo_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA723345 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_wo_outlying" + save_PRJNA723345,
)


##########################################################
##### train Density preserving supervised UMAP model #####
##########################################################
reducer_by_PRJNA722418 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_by_PRJNA722418.fit(X1_PRJNA722418, membership)

X1_PRJNA722418_densmap_supervised = reducer_by_PRJNA722418.transform(X1_PRJNA722418)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    X1_PRJNA722418_densmap_supervised[:, 0],
    X1_PRJNA722418_densmap_supervised[:, 1],
    c=membership,
    cmap="nipy_spectral",
    edgecolor="k",
    label=membership,
)
plt.colorbar(boundaries=np.arange(64) - 0.5).set_ticks(np.arange(63))
plt.savefig(
    os.path.join(
        PLOTS_DIR,
        f"astrocytes_X1_PRJNA722418-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-densmap_supervised.pdf",
    )
)


# Fit linear separator using X_train
svm_PRJNA722418 = SVC(kernel="linear", C=100)
svm_PRJNA722418.fit(X1_PRJNA722418, membership)
joblib.dump(
    svm_PRJNA722418,
    f"models/svm/PRJNA722418-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pickle",
)

# Generate labels for X_test
predictions_X2_PRJNA722418 = svm_PRJNA722418.predict(X2_PRJNA722418)
predictions_X3_PRJNA722418 = svm_PRJNA722418.predict(X3_PRJNA722418)


PRJNA722418.obs["clusters"] = PRJNA722418.obs["condit"]
PRJNA722418.obs["clusters"][
    PRJNA722418.obs["train"] & (PRJNA722418.obs["bioproject"] == "PRJNA779749")
] = membership
PRJNA722418.obs["clusters"][
    PRJNA722418.obs["test"] & (PRJNA722418.obs["bioproject"] == "PRJNA779749")
] = predictions_X2_PRJNA722418
PRJNA722418.obs["clusters"][
    PRJNA722418.obs["bioproject"] == "PRJNA722418"
] = predictions_X3_PRJNA722418
PRJNA722418.obs["clusters"] = pd.Categorical(
        values=PRJNA722418.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA722418.obs["clusters"].astype(str)))),
    )
PRJNA722418_counts.obs["clusters"] = predictions_X3_PRJNA722418
PRJNA722418_counts.obs["clusters"] = pd.Categorical(
        values=PRJNA722418_counts.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA722418_counts.obs["clusters"].astype(str)))),
    )


reducer_PRJNA722418 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_PRJNA722418.fit(PRJNA722418.obsm["X_pca_harmony"], PRJNA722418.obs["clusters"].astype(str))
PRJNA722418.obsm["X_umap"] = reducer_PRJNA722418.transform(
    PRJNA722418.obsm["X_pca_harmony"]
)

save_PRJNA722418 = f"-supervised-PRJNA722418-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.embedding(
    PRJNA722418,
    basis="X_pacmap",
    color=["project", "clusters"],
    title=f"PaCMAP: paired integration, PRJNA722418 (amb.FPR={cb_fpr}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save=save_PRJNA722418,
)
print(PRJNA722418.obs['clusters'].value_counts().sort_values())


sc.pl.umap(
    PRJNA722418,
    color=["project"],
    title=f"UMAP: supervised integration, PRJNA722418 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-project" + save_PRJNA722418,
)

sc.tl.embedding_density(PRJNA722418, basis='umap', groupby='project')
sc.pl.embedding_density(
    PRJNA722418,
    basis='umap',
    key='umap_density_project',
    save=save_PRJNA722418)


sc.pl.umap(
    PRJNA722418,
    color=["clusters"],
    title=f"UMAP: supervised integration, PRJNA722418 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-project" + save_PRJNA722418,
)

sc.tl.embedding_density(PRJNA722418, basis='umap', groupby='clusters')
sc.pl.embedding_density(
    PRJNA722418,
    basis='umap',
    key='umap_density_clusters',
    save=save_PRJNA722418)


PRJNA722418.write(
    f"data/resolved_subregions/paired_mtx-PRJNA779749_and_PRJNA722418-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)

PRJNA722418_counts = update_subregion_embedding(PRJNA722418_counts)
PRJNA722418_counts.write(
    f"data/resolved_subregions/PRJNA722418-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA722418_counts.write_loom(
    f"data/resolved_subregions/PRJNA722418-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.loom"
)
pd.DataFrame(
    PRJNA722418_counts.obsm["X_umap"], index=PRJNA722418_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA722418-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-umap.tsv",
    sep="\t",
    header=True,
)
pd.DataFrame(
    PRJNA722418_counts.obsm["X_pacmap"], index=PRJNA722418_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA722418-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-pacmap.tsv",
    sep="\t",
    header=True,
)


sc.pl.umap(
    PRJNA722418,
    color=[g for g in npep if g in PRJNA722418.var_names],
    frameon=False,
    show=False,
    save="-npep" + save_PRJNA722418,
)


sc.pl.umap(
    PRJNA722418,
    color=[g for g in npr if g in PRJNA722418.var_names],
    frameon=False,
    show=False,
    save="-npr" + save_PRJNA722418,
)

sc.pl.umap(
    PRJNA722418,
    color=[g for g in nmr if g in PRJNA722418.var_names],
    frameon=False,
    show=False,
    save="-nmr" + save_PRJNA722418,
)


sc.pl.umap(
    PRJNA722418,
    color=[g for g in genes_embed if g in PRJNA722418.var_names],
    frameon=False,
    show=False,
    save="-adgen" + save_PRJNA722418,
)


PRJNA722418.obs["learning"] = pd.Categorical(
    ["train" if x else "test" for x in PRJNA722418.obs["train"]]
)
outlier_scores = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=ks, contamination=0.1
).fit_predict(PRJNA722418.obsm["X_umap"])


PRJNA722418_outlying = PRJNA722418[outlier_scores == -1]
PRJNA722418_outlying.shape


PRJNA722418_outlying


sc.pl.umap(
    PRJNA722418_outlying,
    color=[g for g in genes_embed if g in PRJNA722418.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save=False,
)


sc.pl.umap(
    PRJNA722418_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA722418 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_outlying" + save_PRJNA722418,
)


sc.pl.umap(
    PRJNA722418_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA722418 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_outlying" + save_PRJNA722418,
)


PRJNA722418_wo_outlying = PRJNA722418[outlier_scores != -1]
PRJNA722418_wo_outlying.shape


sc.pl.umap(
    PRJNA722418_wo_outlying,
    color=[g for g in genes_embed if g in PRJNA722418.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgene_wo_outlying" + save_PRJNA722418,
)


sc.pl.umap(
    PRJNA722418_wo_outlying,
    color=[g for g in npep if g in PRJNA722418.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep_wo_outlying" + save_PRJNA722418,
)


sc.pl.umap(
    PRJNA722418_wo_outlying,
    color=[g for g in npr if g in PRJNA722418.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr_wo_outlying" + save_PRJNA722418,
)


sc.pl.umap(
    PRJNA722418_wo_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA722418 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_wo_outlying" + save_PRJNA722418,
)


sc.pl.umap(
    PRJNA722418_wo_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA722418 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_wo_outlying" + save_PRJNA722418,
)


##########################################################
##### train Density preserving supervised UMAP model #####
##########################################################
reducer_by_PRJNA705596 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_by_PRJNA705596.fit(X1_PRJNA705596, membership)

X1_PRJNA705596_densmap_supervised = reducer_by_PRJNA705596.transform(X1_PRJNA705596)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    X1_PRJNA705596_densmap_supervised[:, 0],
    X1_PRJNA705596_densmap_supervised[:, 1],
    c=membership,
    cmap="nipy_spectral",
    edgecolor="k",
    label=membership,
)
plt.colorbar(boundaries=np.arange(64) - 0.5).set_ticks(np.arange(63))
plt.savefig(
    os.path.join(
        PLOTS_DIR,
        f"astrocytes_X1_PRJNA705596-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-densmap_supervised.pdf",
    )
)


# Fit linear separator using X_train
svm_PRJNA705596 = SVC(kernel="linear", C=100)
svm_PRJNA705596.fit(X1_PRJNA705596, membership)
joblib.dump(
    svm_PRJNA705596,
    f"models/svm/PRJNA705596-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pickle",
)

# Generate labels for X_test
predictions_X2_PRJNA705596 = svm_PRJNA705596.predict(X2_PRJNA705596)
predictions_X3_PRJNA705596 = svm_PRJNA705596.predict(X3_PRJNA705596)


PRJNA705596.obs["clusters"] = PRJNA705596.obs["condit"]
PRJNA705596.obs["clusters"][
    PRJNA705596.obs["train"] & (PRJNA705596.obs["bioproject"] == "PRJNA779749")
] = membership
PRJNA705596.obs["clusters"][
    PRJNA705596.obs["test"] & (PRJNA705596.obs["bioproject"] == "PRJNA779749")
] = predictions_X2_PRJNA705596
PRJNA705596.obs["clusters"][
    PRJNA705596.obs["bioproject"] == "PRJNA705596"
] = predictions_X3_PRJNA705596
PRJNA705596.obs["clusters"] = pd.Categorical(
        values=PRJNA705596.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA705596.obs["clusters"].astype(str)))),
    )
PRJNA705596_counts.obs["clusters"] = predictions_X3_PRJNA705596
PRJNA705596_counts.obs["clusters"] = pd.Categorical(
        values=PRJNA705596_counts.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA705596_counts.obs["clusters"].astype(str)))),
    )


reducer_PRJNA705596 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_PRJNA705596.fit(PRJNA705596.obsm["X_pca_harmony"], PRJNA705596.obs["clusters"].astype(str))
PRJNA705596.obsm["X_umap"] = reducer_PRJNA705596.transform(
    PRJNA705596.obsm["X_pca_harmony"]
)

save_PRJNA705596 = f"-supervised-PRJNA705596-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.embedding(
    PRJNA705596,
    basis="X_pacmap",
    color=["project", "clusters"],
    title=f"PaCMAP: paired integration, PRJNA705596 (amb.FPR={cb_fpr}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save=save_PRJNA705596,
)
print(PRJNA705596.obs['clusters'].value_counts().sort_values())


sc.pl.umap(
    PRJNA705596,
    color=["project"],
    title=f"UMAP: supervised integration, PRJNA705596 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-project" + save_PRJNA705596,
)

sc.tl.embedding_density(PRJNA705596, basis='umap', groupby='project')
sc.pl.embedding_density(
    PRJNA705596,
    basis='umap',
    key='umap_density_project',
    save=save_PRJNA705596)


sc.pl.umap(
    PRJNA705596,
    color=["clusters"],
    title=f"UMAP: supervised integration, PRJNA705596 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    legend_loc="on data",
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-clusters" + save_PRJNA705596,
)

sc.tl.embedding_density(PRJNA705596, basis='umap', groupby='clusters')
sc.pl.embedding_density(
    PRJNA705596,
    basis='umap',
    key='umap_density_clusters',
    save=save_PRJNA705596)


PRJNA705596.write(
    f"data/resolved_subregions/paired_mtx-PRJNA779749_and_PRJNA705596-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)

PRJNA705596_counts = update_subregion_embedding(PRJNA705596_counts)
PRJNA705596_counts.write(
    f"data/resolved_subregions/PRJNA705596-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA705596_counts.write_loom(
    f"data/resolved_subregions/PRJNA705596-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.loom"
)
pd.DataFrame(
    PRJNA705596_counts.obsm["X_umap"], index=PRJNA705596_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA705596-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-umap.tsv",
    sep="\t",
    header=True,
)
pd.DataFrame(
    PRJNA705596_counts.obsm["X_pacmap"], index=PRJNA705596_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA705596-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-pacmap.tsv",
    sep="\t",
    header=True,
)


sc.pl.umap(
    PRJNA705596,
    color=[g for g in npep if g in PRJNA705596.var_names],
    frameon=False,
    show=False,
    save="-npep" + save_PRJNA705596,
)


sc.pl.umap(
    PRJNA705596,
    color=[g for g in npr if g in PRJNA705596.var_names],
    frameon=False,
    show=False,
    save="-npr" + save_PRJNA705596,
)

sc.pl.umap(
    PRJNA705596,
    color=[g for g in nmr if g in PRJNA705596.var_names],
    frameon=False,
    show=False,
    save="-nmr" + save_PRJNA705596,
)



sc.pl.umap(
    PRJNA705596,
    color=[g for g in genes_embed if g in PRJNA705596.var_names],
    frameon=False,
    show=False,
    save="-adgen" + save_PRJNA705596,
)


PRJNA705596.obs["learning"] = pd.Categorical(
    ["train" if x else "test" for x in PRJNA705596.obs["train"]]
)
outlier_scores = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=ks, contamination=0.1
).fit_predict(PRJNA705596.obsm["X_umap"])


PRJNA705596_outlying = PRJNA705596[outlier_scores == -1]
PRJNA705596_outlying.shape


PRJNA705596_outlying


sc.pl.umap(
    PRJNA705596_outlying,
    color=[g for g in genes_embed if g in PRJNA705596.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save=False,
)


sc.pl.umap(
    PRJNA705596_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA705596 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_outlying" + save_PRJNA705596,
)


sc.pl.umap(
    PRJNA705596_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA705596 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_outlying" + save_PRJNA705596,
)


PRJNA705596_wo_outlying = PRJNA705596[outlier_scores != -1]
PRJNA705596_wo_outlying.shape


sc.pl.umap(
    PRJNA705596_wo_outlying,
    color=[g for g in genes_embed if g in PRJNA705596.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgene_wo_outlying" + save_PRJNA705596,
)


sc.pl.umap(
    PRJNA705596_wo_outlying,
    color=[g for g in npep if g in PRJNA705596.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep_wo_outlying" + save_PRJNA705596,
)


sc.pl.umap(
    PRJNA705596_wo_outlying,
    color=[g for g in npr if g in PRJNA705596.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr_wo_outlying" + save_PRJNA705596,
)


sc.pl.umap(
    PRJNA705596_wo_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA705596 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_wo_outlying" + save_PRJNA705596,
)


sc.pl.umap(
    PRJNA705596_wo_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA705596 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_wo_outlying" + save_PRJNA705596,
)


##########################################################
##### train Density preserving supervised UMAP model #####
##########################################################
reducer_by_PRJNA679294 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_by_PRJNA679294.fit(X1_PRJNA679294, membership)

X1_PRJNA679294_densmap_supervised = reducer_by_PRJNA679294.transform(X1_PRJNA679294)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    X1_PRJNA679294_densmap_supervised[:, 0],
    X1_PRJNA679294_densmap_supervised[:, 1],
    c=membership,
    cmap="nipy_spectral",
    edgecolor="k",
    label=membership,
)
plt.colorbar(boundaries=np.arange(64) - 0.5).set_ticks(np.arange(63))
plt.savefig(
    os.path.join(
        PLOTS_DIR,
        f"astrocytes_X1_PRJNA679294-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-densmap_supervised.pdf",
    )
)


# Fit linear separator using X_train
svm_PRJNA679294 = SVC(kernel="linear", C=100)
svm_PRJNA679294.fit(X1_PRJNA679294, membership)
joblib.dump(
    svm_PRJNA679294,
    f"models/svm/PRJNA679294-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pickle",
)

# Generate labels for X_test
predictions_X2_PRJNA679294 = svm_PRJNA679294.predict(X2_PRJNA679294)
predictions_X3_PRJNA679294 = svm_PRJNA679294.predict(X3_PRJNA679294)


PRJNA679294.obs["clusters"] = PRJNA679294.obs["condit"]
PRJNA679294.obs["clusters"][
    PRJNA679294.obs["train"] & (PRJNA679294.obs["bioproject"] == "PRJNA779749")
] = membership
PRJNA679294.obs["clusters"][
    PRJNA679294.obs["test"] & (PRJNA679294.obs["bioproject"] == "PRJNA779749")
] = predictions_X2_PRJNA679294
PRJNA679294.obs["clusters"][
    PRJNA679294.obs["bioproject"] == "PRJNA679294"
] = predictions_X3_PRJNA679294
PRJNA679294.obs["clusters"] = pd.Categorical(
        values=PRJNA679294.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA679294.obs["clusters"].astype(str)))),
    )
PRJNA679294_counts.obs["clusters"] = predictions_X3_PRJNA679294
PRJNA679294_counts.obs["clusters"] = pd.Categorical(
        values=PRJNA679294_counts.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA679294_counts.obs["clusters"].astype(str)))),
    )


reducer_PRJNA679294 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_PRJNA679294.fit(PRJNA679294.obsm["X_pca_harmony"], PRJNA679294.obs["clusters"].astype(str))
PRJNA679294.obsm["X_umap"] = reducer_PRJNA679294.transform(
    PRJNA679294.obsm["X_pca_harmony"]
)

save_PRJNA679294 = f"-supervised-PRJNA679294-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.embedding(
    PRJNA679294,
    basis="X_pacmap",
    color=["project", "clusters"],
    title=f"PaCMAP: paired integration, PRJNA679294 (amb.FPR={cb_fpr}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save=save_PRJNA679294,
)
print(PRJNA679294.obs['clusters'].value_counts().sort_values())


sc.pl.umap(
    PRJNA679294,
    color=["project"],
    title=f"UMAP: supervised integration, PRJNA679294 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-project" + save_PRJNA679294,
)

sc.tl.embedding_density(PRJNA679294, basis='umap', groupby='project')
sc.pl.embedding_density(
    PRJNA679294,
    basis='umap',
    key='umap_density_project',
    save=save_PRJNA679294)


sc.pl.umap(
    PRJNA679294,
    color=["clusters"],
    title=f"UMAP: supervised integration, PRJNA679294 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-clusters" + save_PRJNA679294,
)

sc.tl.embedding_density(PRJNA679294, basis='umap', groupby='clusters')
sc.pl.embedding_density(
    PRJNA679294,
    basis='umap',
    key='umap_density_clusters',
    save=save_PRJNA679294)


PRJNA679294.write(
    f"data/resolved_subregions/paired_mtx-PRJNA779749_and_PRJNA679294-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)

PRJNA679294_counts = update_subregion_embedding(PRJNA679294_counts)
PRJNA679294_counts.write(
    f"data/resolved_subregions/PRJNA679294-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA679294_counts.write_loom(
    f"data/resolved_subregions/PRJNA679294-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.loom"
)
pd.DataFrame(
    PRJNA679294_counts.obsm["X_umap"], index=PRJNA679294_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA679294-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-umap.tsv",
    sep="\t",
    header=True,
)
pd.DataFrame(
    PRJNA679294_counts.obsm["X_pacmap"], index=PRJNA679294_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA679294-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-pacmap.tsv",
    sep="\t",
    header=True,
)


sc.pl.umap(
    PRJNA679294,
    color=[g for g in npep if g in PRJNA679294.var_names],
    frameon=False,
    show=False,
    save="-npep" + save_PRJNA679294,
)


sc.pl.umap(
    PRJNA679294,
    color=[g for g in npr if g in PRJNA679294.var_names],
    frameon=False,
    show=False,
    save="-npr" + save_PRJNA679294,
)

sc.pl.umap(
    PRJNA679294,
    color=[g for g in nmr if g in PRJNA679294.var_names],
    frameon=False,
    show=False,
    save="-nmr" + save_PRJNA679294,
)


sc.pl.umap(
    PRJNA679294,
    color=[g for g in genes_embed if g in PRJNA679294.var_names],
    frameon=False,
    show=False,
    save="-adgen" + save_PRJNA679294,
)


PRJNA679294.obs["learning"] = pd.Categorical(
    ["train" if x else "test" for x in PRJNA679294.obs["train"]]
)
outlier_scores = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=ks, contamination=0.1
).fit_predict(PRJNA679294.obsm["X_umap"])


PRJNA679294_outlying = PRJNA679294[outlier_scores == -1]
PRJNA679294_outlying.shape


PRJNA679294_outlying


sc.pl.umap(
    PRJNA679294_outlying,
    color=[g for g in genes_embed if g in PRJNA679294.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save=False,
)


sc.pl.umap(
    PRJNA679294_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA679294 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_outlying" + save_PRJNA679294,
)


sc.pl.umap(
    PRJNA679294_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA679294 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_outlying" + save_PRJNA679294,
)


PRJNA679294_wo_outlying = PRJNA679294[outlier_scores != -1]
PRJNA679294_wo_outlying.shape


sc.pl.umap(
    PRJNA679294_wo_outlying,
    color=[g for g in genes_embed if g in PRJNA679294.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgene_wo_outlying" + save_PRJNA679294,
)


sc.pl.umap(
    PRJNA679294_wo_outlying,
    color=[g for g in npep if g in PRJNA679294.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep_wo_outlying" + save_PRJNA679294,
)


sc.pl.umap(
    PRJNA679294_wo_outlying,
    color=[g for g in npr if g in PRJNA679294.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr_wo_outlying" + save_PRJNA679294,
)


sc.pl.umap(
    PRJNA679294_wo_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA679294 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_wo_outlying" + save_PRJNA679294,
)


sc.pl.umap(
    PRJNA679294_wo_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA679294 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_wo_outlying" + save_PRJNA679294,
)


##########################################################
##### train Density preserving supervised UMAP model #####
##########################################################
reducer_by_PRJNA611624 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_by_PRJNA611624.fit(X1_PRJNA611624, membership)

X1_PRJNA611624_densmap_supervised = reducer_by_PRJNA611624.transform(X1_PRJNA611624)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    X1_PRJNA611624_densmap_supervised[:, 0],
    X1_PRJNA611624_densmap_supervised[:, 1],
    c=membership,
    cmap="nipy_spectral",
    edgecolor="k",
    label=membership,
)
plt.colorbar(boundaries=np.arange(64) - 0.5).set_ticks(np.arange(63))
plt.savefig(
    os.path.join(
        PLOTS_DIR,
        f"astrocytes_X1_PRJNA611624-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-densmap_supervised.pdf",
    )
)


# Fit linear separator using X_train
svm_PRJNA611624 = SVC(kernel="linear", C=100)
svm_PRJNA611624.fit(X1_PRJNA611624, membership)
joblib.dump(
    svm_PRJNA611624,
    f"models/svm/PRJNA611624-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pickle",
)

# Generate labels for X_test
predictions_X2_PRJNA611624 = svm_PRJNA611624.predict(X2_PRJNA611624)
predictions_X3_PRJNA611624 = svm_PRJNA611624.predict(X3_PRJNA611624)


PRJNA611624.obs["clusters"] = PRJNA611624.obs["condit"]
PRJNA611624.obs["clusters"][
    PRJNA611624.obs["train"] & (PRJNA611624.obs["bioproject"] == "PRJNA779749")
] = membership
PRJNA611624.obs["clusters"][
    PRJNA611624.obs["test"] & (PRJNA611624.obs["bioproject"] == "PRJNA779749")
] = predictions_X2_PRJNA611624
PRJNA611624.obs["clusters"][
    PRJNA611624.obs["bioproject"] == "PRJNA611624"
] = predictions_X3_PRJNA611624
PRJNA611624.obs["clusters"] = pd.Categorical(
        values=PRJNA611624.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA611624.obs["clusters"].astype(str)))),
    )
PRJNA611624_counts.obs["clusters"] = predictions_X3_PRJNA611624
PRJNA611624_counts.obs["clusters"] = pd.Categorical(
        values=PRJNA611624_counts.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA611624_counts.obs["clusters"].astype(str)))),
    )


reducer_PRJNA611624 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_PRJNA611624.fit(PRJNA611624.obsm["X_pca_harmony"], PRJNA611624.obs["clusters"].astype(str))
PRJNA611624.obsm["X_umap"] = reducer_PRJNA611624.transform(
    PRJNA611624.obsm["X_pca_harmony"]
)

save_PRJNA611624 = f"-supervised-PRJNA611624-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.embedding(
    PRJNA611624,
    basis="X_pacmap",
    color=["project", "clusters"],
    title=f"PaCMAP: paired integration, PRJNA611624 (amb.FPR={cb_fpr}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save=save_PRJNA611624,
)
print(PRJNA611624.obs['clusters'].value_counts().sort_values())


sc.pl.umap(
    PRJNA611624,
    color=["project"],
    title=f"UMAP: supervised integration, PRJNA611624 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-project" + save_PRJNA611624,
)

sc.tl.embedding_density(PRJNA611624, basis='umap', groupby='project')
sc.pl.embedding_density(
    PRJNA611624,
    basis='umap',
    key='umap_density_project',
    save=save_PRJNA611624)


sc.pl.umap(
    PRJNA611624,
    color=["clusters"],
    title=f"UMAP: supervised integration, PRJNA611624 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-clusters" + save_PRJNA611624,
)

sc.tl.embedding_density(PRJNA611624, basis='umap', groupby='clusters')
sc.pl.embedding_density(
    PRJNA611624,
    basis='umap',
    key='umap_density_clusters',
    save=save_PRJNA611624)


PRJNA611624.write(
    f"data/resolved_subregions/paired_mtx-PRJNA779749_and_PRJNA611624-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)

PRJNA611624_counts = update_subregion_embedding(PRJNA611624_counts)
PRJNA611624_counts.write(
    f"data/resolved_subregions/PRJNA611624-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA611624_counts.write_loom(
    f"data/resolved_subregions/PRJNA611624-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.loom"
)
pd.DataFrame(
    PRJNA611624_counts.obsm["X_umap"], index=PRJNA611624_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA611624-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-umap.tsv",
    sep="\t",
    header=True,
)
pd.DataFrame(
    PRJNA611624_counts.obsm["X_pacmap"], index=PRJNA611624_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA611624-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-pacmap.tsv",
    sep="\t",
    header=True,
)


sc.pl.umap(
    PRJNA611624,
    color=[g for g in npep if g in PRJNA611624.var_names],
    frameon=False,
    show=False,
    save="-npep" + save_PRJNA611624,
)


sc.pl.umap(
    PRJNA611624,
    color=[g for g in npr if g in PRJNA611624.var_names],
    frameon=False,
    show=False,
    save="-npr" + save_PRJNA611624,
)


sc.pl.umap(
    PRJNA611624,
    color=[g for g in nmr if g in PRJNA611624.var_names],
    frameon=False,
    show=False,
    save="-nmr" + save_PRJNA611624,
)


sc.pl.umap(
    PRJNA611624,
    color=[g for g in genes_embed if g in PRJNA611624.var_names],
    frameon=False,
    show=False,
    save="-adgen" + save_PRJNA611624,
)


PRJNA611624.obs["learning"] = pd.Categorical(
    ["train" if x else "test" for x in PRJNA611624.obs["train"]]
)
outlier_scores = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=ks, contamination=0.1
).fit_predict(PRJNA611624.obsm["X_umap"])


PRJNA611624_outlying = PRJNA611624[outlier_scores == -1]
PRJNA611624_outlying.shape


PRJNA611624_outlying


sc.pl.umap(
    PRJNA611624_outlying,
    color=[g for g in genes_embed if g in PRJNA611624.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save=False,
)


sc.pl.umap(
    PRJNA611624_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA611624 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_outlying" + save_PRJNA611624,
)


sc.pl.umap(
    PRJNA611624_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA611624 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_outlying" + save_PRJNA611624,
)


PRJNA611624_wo_outlying = PRJNA611624[outlier_scores != -1]
PRJNA611624_wo_outlying.shape


sc.pl.umap(
    PRJNA611624_wo_outlying,
    color=[g for g in genes_embed if g in PRJNA611624.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgene_wo_outlying" + save_PRJNA611624,
)


sc.pl.umap(
    PRJNA611624_wo_outlying,
    color=[g for g in npep if g in PRJNA611624.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep_wo_outlying" + save_PRJNA611624,
)


sc.pl.umap(
    PRJNA611624_wo_outlying,
    color=[g for g in npr if g in PRJNA611624.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr_wo_outlying" + save_PRJNA611624,
)


sc.pl.umap(
    PRJNA611624_wo_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA611624 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_wo_outlying" + save_PRJNA611624,
)


sc.pl.umap(
    PRJNA611624_wo_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA611624 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_wo_outlying" + save_PRJNA611624,
)


##########################################################
##### train Density preserving supervised UMAP model #####
##########################################################
reducer_by_PRJNA604055 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_by_PRJNA604055.fit(X1_PRJNA604055, membership)

X1_PRJNA604055_densmap_supervised = reducer_by_PRJNA604055.transform(X1_PRJNA604055)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    X1_PRJNA604055_densmap_supervised[:, 0],
    X1_PRJNA604055_densmap_supervised[:, 1],
    c=membership,
    cmap="nipy_spectral",
    edgecolor="k",
    label=membership,
)
plt.colorbar(boundaries=np.arange(64) - 0.5).set_ticks(np.arange(63))
plt.savefig(
    os.path.join(
        PLOTS_DIR,
        f"astrocytes_X1_PRJNA604055-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-densmap_supervised.pdf",
    )
)


# Fit linear separator using X_train
svm_PRJNA604055 = SVC(kernel="linear", C=100)
svm_PRJNA604055.fit(X1_PRJNA604055, membership)
joblib.dump(
    svm_PRJNA604055,
    f"models/svm/PRJNA604055-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pickle",
)

# Generate labels for X_test
predictions_X2_PRJNA604055 = svm_PRJNA604055.predict(X2_PRJNA604055)
predictions_X3_PRJNA604055 = svm_PRJNA604055.predict(X3_PRJNA604055)


PRJNA604055.obs["clusters"] = PRJNA604055.obs["condit"]
PRJNA604055.obs["clusters"][
    PRJNA604055.obs["train"] & (PRJNA604055.obs["bioproject"] == "PRJNA779749")
] = membership
PRJNA604055.obs["clusters"][
    PRJNA604055.obs["test"] & (PRJNA604055.obs["bioproject"] == "PRJNA779749")
] = predictions_X2_PRJNA604055
PRJNA604055.obs["clusters"][
    PRJNA604055.obs["bioproject"] == "PRJNA604055"
] = predictions_X3_PRJNA604055
PRJNA604055.obs["clusters"] = pd.Categorical(
        values=PRJNA604055.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA604055.obs["clusters"].astype(str)))),
    )
PRJNA604055_counts.obs["clusters"] = predictions_X3_PRJNA604055
PRJNA604055_counts.obs["clusters"] = pd.Categorical(
        values=PRJNA604055_counts.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA604055_counts.obs["clusters"].astype(str)))),
    )


reducer_PRJNA604055 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_PRJNA604055.fit(PRJNA604055.obsm["X_pca_harmony"], PRJNA604055.obs["clusters"].astype(str))
PRJNA604055.obsm["X_umap"] = reducer_PRJNA604055.transform(
    PRJNA604055.obsm["X_pca_harmony"]
)

save_PRJNA604055 = f"-supervised-PRJNA604055-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.embedding(
    PRJNA604055,
    basis="X_pacmap",
    color=["project", "clusters"],
    title=f"PaCMAP: paired integration, PRJNA604055 (amb.FPR={cb_fpr}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save=save_PRJNA604055,
)
print(PRJNA604055.obs['clusters'].value_counts().sort_values())


sc.pl.umap(
    PRJNA604055,
    color=["project"],
    title=f"UMAP: supervised integration, PRJNA604055 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-project" + save_PRJNA604055,
)

sc.tl.embedding_density(PRJNA604055, basis='umap', groupby='project')
sc.pl.embedding_density(
    PRJNA604055,
    basis='umap',
    key='umap_density_project',
    save=save_PRJNA604055)


sc.pl.umap(
    PRJNA604055,
    color=["clusters"],
    title=f"UMAP: supervised integration, PRJNA604055 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-clusters" + save_PRJNA604055,
)

sc.tl.embedding_density(PRJNA604055, basis='umap', groupby='clusters')
sc.pl.embedding_density(
    PRJNA604055,
    basis='umap',
    key='umap_density_clusters',
    save=save_PRJNA604055)


PRJNA604055.write(
    f"data/resolved_subregions/paired_mtx-PRJNA779749_and_PRJNA604055-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)

PRJNA604055_counts = update_subregion_embedding(PRJNA604055_counts)
PRJNA604055_counts.write(
    f"data/resolved_subregions/PRJNA604055-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA604055_counts.write_loom(
    f"data/resolved_subregions/PRJNA604055-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.loom"
)
pd.DataFrame(
    PRJNA604055_counts.obsm["X_umap"], index=PRJNA604055_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA604055-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-umap.tsv",
    sep="\t",
    header=True,
)
pd.DataFrame(
    PRJNA604055_counts.obsm["X_pacmap"], index=PRJNA604055_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA604055-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-pacmap.tsv",
    sep="\t",
    header=True,
)


sc.pl.umap(
    PRJNA604055,
    color=[g for g in npep if g in PRJNA604055.var_names],
    frameon=False,
    show=False,
    save="-npep" + save_PRJNA604055,
)


sc.pl.umap(
    PRJNA604055,
    color=[g for g in npr if g in PRJNA604055.var_names],
    frameon=False,
    show=False,
    save="-npr" + save_PRJNA604055,
)


sc.pl.umap(
    PRJNA604055,
    color=[g for g in nmr if g in PRJNA604055.var_names],
    frameon=False,
    show=False,
    save="-nmr" + save_PRJNA604055,
)


sc.pl.umap(
    PRJNA604055,
    color=[g for g in genes_embed if g in PRJNA604055.var_names],
    frameon=False,
    show=False,
    save="-adgen" + save_PRJNA604055,
)


PRJNA604055.obs["learning"] = pd.Categorical(
    ["train" if x else "test" for x in PRJNA604055.obs["train"]]
)
outlier_scores = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=ks, contamination=0.1
).fit_predict(PRJNA604055.obsm["X_umap"])


PRJNA604055_outlying = PRJNA604055[outlier_scores == -1]
PRJNA604055_outlying.shape


PRJNA604055_outlying


sc.pl.umap(
    PRJNA604055_outlying,
    color=[g for g in genes_embed if g in PRJNA604055.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save=False,
)


sc.pl.umap(
    PRJNA604055_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA604055 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_outlying" + save_PRJNA604055,
)


sc.pl.umap(
    PRJNA604055_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA604055 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_outlying" + save_PRJNA604055,
)


PRJNA604055_wo_outlying = PRJNA604055[outlier_scores != -1]
PRJNA604055_wo_outlying.shape


sc.pl.umap(
    PRJNA604055_wo_outlying,
    color=[g for g in genes_embed if g in PRJNA604055.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgene_wo_outlying" + save_PRJNA604055,
)


sc.pl.umap(
    PRJNA604055_wo_outlying,
    color=[g for g in npep if g in PRJNA604055.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep_wo_outlying" + save_PRJNA604055,
)


sc.pl.umap(
    PRJNA604055_wo_outlying,
    color=[g for g in npr if g in PRJNA604055.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr_wo_outlying" + save_PRJNA604055,
)


sc.pl.umap(
    PRJNA604055_wo_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA604055 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_wo_outlying" + save_PRJNA604055,
)


sc.pl.umap(
    PRJNA604055_wo_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA604055 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_wo_outlying" + save_PRJNA604055,
)


##########################################################
##### train Density preserving supervised UMAP model #####
##########################################################
reducer_by_PRJNA548532 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_by_PRJNA548532.fit(X1_PRJNA548532, membership)

X1_PRJNA548532_densmap_supervised = reducer_by_PRJNA548532.transform(X1_PRJNA548532)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    X1_PRJNA548532_densmap_supervised[:, 0],
    X1_PRJNA548532_densmap_supervised[:, 1],
    c=membership,
    cmap="nipy_spectral",
    edgecolor="k",
    label=membership,
)
plt.colorbar(boundaries=np.arange(64) - 0.5).set_ticks(np.arange(63))
plt.savefig(
    os.path.join(
        PLOTS_DIR,
        f"astrocytes_X1_PRJNA548532-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-densmap_supervised.pdf",
    )
)


# Fit linear separator using X_train
svm_PRJNA548532 = SVC(kernel="linear", C=100)
svm_PRJNA548532.fit(X1_PRJNA548532, membership)
joblib.dump(
    svm_PRJNA548532,
    f"models/svm/PRJNA548532-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pickle",
)

# Generate labels for X_test
predictions_X2_PRJNA548532 = svm_PRJNA548532.predict(X2_PRJNA548532)
predictions_X3_PRJNA548532 = svm_PRJNA548532.predict(X3_PRJNA548532)


PRJNA548532.obs["clusters"] = PRJNA548532.obs["condit"]
PRJNA548532.obs["clusters"][
    PRJNA548532.obs["train"] & (PRJNA548532.obs["bioproject"] == "PRJNA779749")
] = membership
PRJNA548532.obs["clusters"][
    PRJNA548532.obs["test"] & (PRJNA548532.obs["bioproject"] == "PRJNA779749")
] = predictions_X2_PRJNA548532
PRJNA548532.obs["clusters"][
    PRJNA548532.obs["bioproject"] == "PRJNA548532"
] = predictions_X3_PRJNA548532
PRJNA548532.obs["clusters"] = pd.Categorical(
        values=PRJNA548532.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA548532.obs["clusters"].astype(str)))),
    )
PRJNA548532_counts.obs["clusters"] = predictions_X3_PRJNA548532
PRJNA548532_counts.obs["clusters"] = pd.Categorical(
        values=PRJNA548532_counts.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA548532_counts.obs["clusters"].astype(str)))),
    )


reducer_PRJNA548532 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_PRJNA548532.fit(PRJNA548532.obsm["X_pca_harmony"], PRJNA548532.obs["clusters"].astype(str))
PRJNA548532.obsm["X_umap"] = reducer_PRJNA548532.transform(
    PRJNA548532.obsm["X_pca_harmony"]
)

save_PRJNA548532 = f"-supervised-PRJNA548532-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.embedding(
    PRJNA548532,
    basis="X_pacmap",
    color=["project", "clusters"],
    title=f"PaCMAP: paired integration, PRJNA548532 (amb.FPR={cb_fpr}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save=save_PRJNA548532,
)
print(PRJNA548532.obs['clusters'].value_counts().sort_values())


sc.pl.umap(
    PRJNA548532,
    color=["project"],
    title=f"UMAP: supervised integration, PRJNA548532 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-project" + save_PRJNA548532,
)

sc.tl.embedding_density(PRJNA548532, basis='umap', groupby='project')
sc.pl.embedding_density(
    PRJNA548532,
    basis='umap',
    key='umap_density_project',
    save=save_PRJNA548532)


sc.pl.umap(
    PRJNA548532,
    color=["clusters"],
    title=f"UMAP: supervised integration, PRJNA548532 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-clusters" + save_PRJNA548532,
)

sc.tl.embedding_density(PRJNA548532, basis='umap', groupby='clusters')
sc.pl.embedding_density(
    PRJNA548532,
    basis='umap',
    key='umap_density_clusters',
    save=save_PRJNA548532)


PRJNA548532.write(
    f"data/resolved_subregions/paired_mtx-PRJNA779749_and_PRJNA548532-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)

PRJNA548532_counts = update_subregion_embedding(PRJNA548532_counts)
PRJNA548532_counts.write(
    f"data/resolved_subregions/PRJNA548532-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA548532_counts.write_loom(
    f"data/resolved_subregions/PRJNA548532-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.loom"
)
pd.DataFrame(
    PRJNA548532_counts.obsm["X_umap"], index=PRJNA548532_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA548532-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-umap.tsv",
    sep="\t",
    header=True,
)
pd.DataFrame(
    PRJNA548532_counts.obsm["X_pacmap"], index=PRJNA548532_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA548532-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-pacmap.tsv",
    sep="\t",
    header=True,
)


sc.pl.umap(
    PRJNA548532,
    color=[g for g in npep if g in PRJNA548532.var_names],
    frameon=False,
    show=False,
    save="-npep" + save_PRJNA548532,
)


sc.pl.umap(
    PRJNA548532,
    color=[g for g in npr if g in PRJNA548532.var_names],
    frameon=False,
    show=False,
    save="-npr" + save_PRJNA548532,
)


sc.pl.umap(
    PRJNA548532,
    color=[g for g in nmr if g in PRJNA548532.var_names],
    frameon=False,
    show=False,
    save="-nmr" + save_PRJNA548532,
)


sc.pl.umap(
    PRJNA548532,
    color=[g for g in genes_embed if g in PRJNA548532.var_names],
    frameon=False,
    show=False,
    save="-adgen" + save_PRJNA548532,
)


PRJNA548532.obs["learning"] = pd.Categorical(
    ["train" if x else "test" for x in PRJNA548532.obs["train"]]
)
outlier_scores = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=ks, contamination=0.1
).fit_predict(PRJNA548532.obsm["X_umap"])


PRJNA548532_outlying = PRJNA548532[outlier_scores == -1]
PRJNA548532_outlying.shape



sc.pl.umap(
    PRJNA548532_outlying,
    color=[g for g in genes_embed if g in PRJNA548532.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save=False,
)


sc.pl.umap(
    PRJNA548532_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA548532 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_outlying" + save_PRJNA548532,
)


sc.pl.umap(
    PRJNA548532_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA548532 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_outlying" + save_PRJNA548532,
)


PRJNA548532_wo_outlying = PRJNA548532[outlier_scores != -1]
PRJNA548532_wo_outlying.shape


sc.pl.umap(
    PRJNA548532_wo_outlying,
    color=[g for g in genes_embed if g in PRJNA548532.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgene_wo_outlying" + save_PRJNA548532,
)


sc.pl.umap(
    PRJNA548532_wo_outlying,
    color=[g for g in npep if g in PRJNA548532.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep_wo_outlying" + save_PRJNA548532,
)


sc.pl.umap(
    PRJNA548532_wo_outlying,
    color=[g for g in npr if g in PRJNA548532.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr_wo_outlying" + save_PRJNA548532,
)


sc.pl.umap(
    PRJNA548532_wo_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA548532 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_wo_outlying" + save_PRJNA548532,
)


sc.pl.umap(
    PRJNA548532_wo_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA548532 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_wo_outlying" + save_PRJNA548532,
)


##########################################################
##### train Density preserving supervised UMAP model #####
##########################################################
reducer_by_PRJNA515063 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_by_PRJNA515063.fit(X1_PRJNA515063, membership)

X1_PRJNA515063_densmap_supervised = reducer_by_PRJNA515063.transform(X1_PRJNA515063)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    X1_PRJNA515063_densmap_supervised[:, 0],
    X1_PRJNA515063_densmap_supervised[:, 1],
    c=membership,
    cmap="nipy_spectral",
    edgecolor="k",
    label=membership,
)
plt.colorbar(boundaries=np.arange(64) - 0.5).set_ticks(np.arange(63))
plt.savefig(
    os.path.join(
        PLOTS_DIR,
        f"astrocytes_X1_PRJNA515063-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-densmap_supervised.pdf",
    )
)


# Fit linear separator using X_train
svm_PRJNA515063 = SVC(kernel="linear", C=100)
svm_PRJNA515063.fit(X1_PRJNA515063, membership)
joblib.dump(
    svm_PRJNA515063,
    f"models/svm/PRJNA515063-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pickle",
)

# Generate labels for X_test
predictions_X2_PRJNA515063 = svm_PRJNA515063.predict(X2_PRJNA515063)
predictions_X3_PRJNA515063 = svm_PRJNA515063.predict(X3_PRJNA515063)


PRJNA515063.obs["clusters"] = PRJNA515063.obs["condit"]
PRJNA515063.obs["clusters"][
    PRJNA515063.obs["train"] & (PRJNA515063.obs["bioproject"] == "PRJNA779749")
] = membership
PRJNA515063.obs["clusters"][
    PRJNA515063.obs["test"] & (PRJNA515063.obs["bioproject"] == "PRJNA779749")
] = predictions_X2_PRJNA515063
PRJNA515063.obs["clusters"][
    PRJNA515063.obs["bioproject"] == "PRJNA515063"
] = predictions_X3_PRJNA515063
PRJNA515063.obs["clusters"] = pd.Categorical(
        values=PRJNA515063.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA515063.obs["clusters"].astype(str)))),
    )
PRJNA515063_counts.obs["clusters"] = predictions_X3_PRJNA515063
PRJNA515063_counts.obs["clusters"] = pd.Categorical(
        values=PRJNA515063_counts.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA515063_counts.obs["clusters"].astype(str)))),
    )


reducer_PRJNA515063 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_PRJNA515063.fit(PRJNA515063.obsm["X_pca_harmony"], PRJNA515063.obs["clusters"].astype(str))
PRJNA515063.obsm["X_umap"] = reducer_PRJNA515063.transform(
    PRJNA515063.obsm["X_pca_harmony"]
)

save_PRJNA515063 = f"-supervised-PRJNA515063-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.embedding(
    PRJNA515063,
    basis="X_pacmap",
    color=["project", "clusters"],
    title=f"PaCMAP: paired integration, PRJNA515063 (amb.FPR={cb_fpr}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save=save_PRJNA515063,
)
print(PRJNA515063.obs['clusters'].value_counts().sort_values())


sc.pl.umap(
    PRJNA515063,
    color=["project"],
    title=f"UMAP: supervised integration, PRJNA515063 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-project" + save_PRJNA515063,
)

sc.tl.embedding_density(PRJNA515063, basis='umap', groupby='project')
sc.pl.embedding_density(
    PRJNA515063,
    basis='umap',
    key='umap_density_project',
    save=save_PRJNA515063)


sc.pl.umap(
    PRJNA515063,
    color=["clusters"],
    title=f"UMAP: supervised integration, PRJNA515063 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-clusters" + save_PRJNA515063,
)

sc.tl.embedding_density(PRJNA515063, basis='umap', groupby='clusters')
sc.pl.embedding_density(
    PRJNA515063,
    basis='umap',
    key='umap_density_clusters',
    save=save_PRJNA515063)


PRJNA515063.write(
    f"data/resolved_subregions/paired_mtx-PRJNA779749_and_PRJNA515063-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)

PRJNA515063_counts = update_subregion_embedding(PRJNA515063_counts)
PRJNA515063_counts.write(
    f"data/resolved_subregions/PRJNA515063-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA515063_counts.write_loom(
    f"data/resolved_subregions/PRJNA515063-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.loom"
)
pd.DataFrame(
    PRJNA515063_counts.obsm["X_umap"], index=PRJNA515063_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA515063-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-umap.tsv",
    sep="\t",
    header=True,
)
pd.DataFrame(
    PRJNA515063_counts.obsm["X_pacmap"], index=PRJNA515063_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA515063-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-pacmap.tsv",
    sep="\t",
    header=True,
)


sc.pl.umap(
    PRJNA515063,
    color=[g for g in npep if g in PRJNA515063.var_names],
    frameon=False,
    show=False,
    save="-npep" + save_PRJNA515063,
)


sc.pl.umap(
    PRJNA515063,
    color=[g for g in npr if g in PRJNA515063.var_names],
    frameon=False,
    show=False,
    save="-npr" + save_PRJNA515063,
)


sc.pl.umap(
    PRJNA515063,
    color=[g for g in nmr if g in PRJNA515063.var_names],
    frameon=False,
    show=False,
    save="-nmr" + save_PRJNA515063,
)


sc.pl.umap(
    PRJNA515063,
    color=[g for g in genes_embed if g in PRJNA515063.var_names],
    frameon=False,
    show=False,
    save="-adgen" + save_PRJNA515063,
)


PRJNA515063.obs["learning"] = pd.Categorical(
    ["train" if x else "test" for x in PRJNA515063.obs["train"]]
)
outlier_scores = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=ks, contamination=0.1
).fit_predict(PRJNA515063.obsm["X_umap"])


PRJNA515063_outlying = PRJNA515063[outlier_scores == -1]
PRJNA515063_outlying.shape


PRJNA515063_outlying


sc.pl.umap(
    PRJNA515063_outlying,
    color=[g for g in genes_embed if g in PRJNA515063.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save=False,
)


sc.pl.umap(
    PRJNA515063_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA515063 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_outlying" + save_PRJNA515063,
)


sc.pl.umap(
    PRJNA515063_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA515063 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_outlying" + save_PRJNA515063,
)


PRJNA515063_wo_outlying = PRJNA515063[outlier_scores != -1]
PRJNA515063_wo_outlying.shape


sc.pl.umap(
    PRJNA515063_wo_outlying,
    color=[g for g in genes_embed if g in PRJNA515063.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgene_wo_outlying" + save_PRJNA515063,
)


sc.pl.umap(
    PRJNA515063_wo_outlying,
    color=[g for g in npep if g in PRJNA515063.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep_wo_outlying" + save_PRJNA515063,
)


sc.pl.umap(
    PRJNA515063_wo_outlying,
    color=[g for g in npr if g in PRJNA515063.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr_wo_outlying" + save_PRJNA515063,
)


sc.pl.umap(
    PRJNA515063_wo_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA515063 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_wo_outlying" + save_PRJNA515063,
)


sc.pl.umap(
    PRJNA515063_wo_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA515063 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_wo_outlying" + save_PRJNA515063,
)


##########################################################
##### train Density preserving supervised UMAP model #####
##########################################################
reducer_by_PRJNA453138 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_by_PRJNA453138.fit(X1_PRJNA453138, membership)

X1_PRJNA453138_densmap_supervised = reducer_by_PRJNA453138.transform(X1_PRJNA453138)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(
    X1_PRJNA453138_densmap_supervised[:, 0],
    X1_PRJNA453138_densmap_supervised[:, 1],
    c=membership,
    cmap="nipy_spectral",
    edgecolor="k",
    label=membership,
)
plt.colorbar(boundaries=np.arange(64) - 0.5).set_ticks(np.arange(63))
plt.savefig(
    os.path.join(
        PLOTS_DIR,
        f"astrocytes_X1_PRJNA453138-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-densmap_supervised.pdf",
    )
)


# Fit linear separator using X_train
svm_PRJNA453138 = SVC(kernel="linear", C=100)
svm_PRJNA453138.fit(X1_PRJNA453138, membership)
joblib.dump(
    svm_PRJNA453138,
    f"models/svm/PRJNA453138-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pickle",
)

# Generate labels for X_test
predictions_X2_PRJNA453138 = svm_PRJNA453138.predict(X2_PRJNA453138)
predictions_X3_PRJNA453138 = svm_PRJNA453138.predict(X3_PRJNA453138)


PRJNA453138.obs["clusters"] = PRJNA453138.obs["condit"]
PRJNA453138.obs["clusters"][
    PRJNA453138.obs["train"] & (PRJNA453138.obs["bioproject"] == "PRJNA779749")
] = membership
PRJNA453138.obs["clusters"][
    PRJNA453138.obs["test"] & (PRJNA453138.obs["bioproject"] == "PRJNA779749")
] = predictions_X2_PRJNA453138
PRJNA453138.obs["clusters"][
    PRJNA453138.obs["bioproject"] == "PRJNA453138"
] = predictions_X3_PRJNA453138
PRJNA453138.obs["clusters"] = pd.Categorical(
        values=PRJNA453138.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA453138.obs["clusters"].astype(str)))),
    )
PRJNA453138_counts.obs["clusters"] = predictions_X3_PRJNA453138
PRJNA453138_counts.obs["clusters"] = pd.Categorical(
        values=PRJNA453138_counts.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA453138_counts.obs["clusters"].astype(str)))),
    )


reducer_PRJNA453138 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_PRJNA453138.fit(PRJNA453138.obsm["X_pca_harmony"], PRJNA453138.obs["clusters"].astype(str))
PRJNA453138.obsm["X_umap"] = reducer_PRJNA453138.transform(
    PRJNA453138.obsm["X_pca_harmony"]
)

save_PRJNA453138 = f"-supervised-PRJNA453138-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.embedding(
    PRJNA453138,
    basis="X_pacmap",
    color=["project", "clusters"],
    title=f"PaCMAP: paired integration, PRJNA453138 (amb.FPR={cb_fpr}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save=save_PRJNA453138,
)
print(PRJNA453138.obs['clusters'].value_counts().sort_values())


sc.pl.umap(
    PRJNA453138,
    color=["project"],
    title=f"UMAP: supervised integration, PRJNA453138 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    show=True,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    save="-project" + save_PRJNA453138,
)

sc.tl.embedding_density(PRJNA453138, basis='umap', groupby='project')
sc.pl.embedding_density(
    PRJNA453138,
    basis='umap',
    key='umap_density_project',
    save=save_PRJNA453138)


sc.pl.umap(
    PRJNA453138,
    color=["clusters"],
    title=f"UMAP: supervised integration, PRJNA453138 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    show=True,
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    save="-clusters" + save_PRJNA453138,
)

sc.tl.embedding_density(PRJNA453138, basis='umap', groupby='clusters')
sc.pl.embedding_density(
    PRJNA453138,
    basis='umap',
    key='umap_density_clusters',
    save=save_PRJNA453138)


PRJNA453138.write(
    f"data/resolved_subregions/paired_mtx-PRJNA779749_and_PRJNA453138-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)

PRJNA453138_counts = update_subregion_embedding(PRJNA453138_counts)
PRJNA453138_counts.write(
    f"data/resolved_subregions/PRJNA453138-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA453138_counts.write_loom(
    f"data/resolved_subregions/PRJNA453138-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.loom"
)
pd.DataFrame(
    PRJNA453138_counts.obsm["X_umap"], index=PRJNA453138_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA453138-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-umap.tsv",
    sep="\t",
    header=True,
)
pd.DataFrame(
    PRJNA453138_counts.obsm["X_pacmap"], index=PRJNA453138_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA453138-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-pacmap.tsv",
    sep="\t",
    header=True,
)


sc.pl.umap(
    PRJNA453138,
    color=[g for g in npep if g in PRJNA453138.var_names],
    frameon=False,
    show=False,
    save="-npep" + save_PRJNA453138,
)


sc.pl.umap(
    PRJNA453138,
    color=[g for g in npr if g in PRJNA453138.var_names],
    frameon=False,
    show=False,
    save="-npr" + save_PRJNA453138,
)


sc.pl.umap(
    PRJNA453138,
    color=[g for g in nmr if g in PRJNA453138.var_names],
    frameon=False,
    show=False,
    save="-nmr" + save_PRJNA453138,
)


sc.pl.umap(
    PRJNA453138,
    color=[g for g in genes_embed if g in PRJNA453138.var_names],
    frameon=False,
    show=False,
    save="-adgen" + save_PRJNA453138,
)


PRJNA453138.obs["learning"] = pd.Categorical(
    ["train" if x else "test" for x in PRJNA453138.obs["train"]]
)
outlier_scores = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=ks, contamination=0.1
).fit_predict(PRJNA453138.obsm["X_umap"])


PRJNA453138_outlying = PRJNA453138[outlier_scores == -1]
PRJNA453138_outlying.shape


PRJNA453138_outlying


sc.pl.umap(
    PRJNA453138_outlying,
    color=[g for g in genes_embed if g in PRJNA453138.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save=False,
)


sc.pl.umap(
    PRJNA453138_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA453138 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_outlying" + save_PRJNA453138,
)


sc.pl.umap(
    PRJNA453138_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA453138 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_outlying" + save_PRJNA453138,
)


PRJNA453138_wo_outlying = PRJNA453138[outlier_scores != -1]
PRJNA453138_wo_outlying.shape


sc.pl.umap(
    PRJNA453138_wo_outlying,
    color=[g for g in genes_embed if g in PRJNA453138.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgene_wo_outlying" + save_PRJNA453138,
)


sc.pl.umap(
    PRJNA453138_wo_outlying,
    color=[g for g in npep if g in PRJNA453138.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep_wo_outlying" + save_PRJNA453138,
)


sc.pl.umap(
    PRJNA453138_wo_outlying,
    color=[g for g in npr if g in PRJNA453138.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr_wo_outlying" + save_PRJNA453138,
)


sc.pl.umap(
    PRJNA453138_wo_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA453138 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_wo_outlying" + save_PRJNA453138,
)


sc.pl.umap(
    PRJNA453138_wo_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA453138 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_wo_outlying" + save_PRJNA453138,
)


X_train, X_test = (
    PRJNA779749_counts[PRJNA779749_counts.obs["train"]].X,
    PRJNA779749_counts[PRJNA779749_counts.obs["test"]].X,
)

# Fit linear separator using X_train
svm_PRJNA779749 = SVC(kernel="linear", C=100)
svm_PRJNA779749.fit(X_train, membership)
joblib.dump(
    svm_PRJNA779749,
    f"models/svm/PRJNA779749_counts-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pickle",
)

# Generate labels for X_test
predictions_PRJNA779749 = svm_PRJNA779749.predict(X_test)


PRJNA779749_counts.obs["clusters"] = PRJNA779749_counts.obs["condit"]
PRJNA779749_counts.obs["clusters"][PRJNA779749_counts.obs["train"]] = membership
PRJNA779749_counts.obs["clusters"][
    PRJNA779749_counts.obs["test"]
] = predictions_PRJNA779749
PRJNA779749_counts.obs["clusters"] = pd.Categorical(
        values=PRJNA779749_counts.obs["clusters"].astype(str),
        categories=natsorted(map(str, np.unique(PRJNA779749_counts.obs["clusters"].astype(str)))),
    )
PRJNA779749_counts.obs["learning"] = pd.Categorical(
    ["train" if x else "test" for x in PRJNA779749_counts.obs["train"]]
)


reducer_PRJNA779749 = umap.UMAP(
    densmap=True, n_components=2, random_state=reseed, verbose=False
)
reducer_PRJNA779749.fit(
    PRJNA779749_counts.obsm["X_pca"], PRJNA779749_counts.obs["clusters"].astype(str)
)
PRJNA779749_counts.obsm["X_umap"] = reducer_PRJNA779749.transform(
    PRJNA779749_counts.obsm["X_pca"]
)
PRJNA779749_counts.write(
    f"data/resolved_subregions/PRJNA779749-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.h5ad"
)
PRJNA779749_counts.write_loom(
    f"data/resolved_subregions/PRJNA779749-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.loom"
)
pd.DataFrame(
    PRJNA779749_counts.obsm["X_umap"], index=PRJNA779749_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA779749-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-umap.tsv",
    sep="\t",
    header=True,
)
pd.DataFrame(
    PRJNA779749_counts.obsm["X_pacmap"], index=PRJNA779749_counts.obs_names
).to_csv(
    f"data/resolved_subregions/PRJNA779749-astrocytes_dataset-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-pacmap.tsv",
    sep="\t",
    header=True,
)

save_PRJNA779749 = f"-supervised-PRJNA779749-astrocytes_datasets-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}.pdf"

sc.pl.embedding(
    PRJNA779749_counts,
    basis="X_pacmap",
    color=["learning", "clusters"],
    title=f"PaCMAP: PRJNA779749 (amb.FPR={cb_fpr}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save=save_PRJNA779749,
)


sc.pl.umap(
    PRJNA779749_counts,
    color=["learning"],
    title=f"UMAP: supervised density, PRJNA779749 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-learning_split" + save_PRJNA779749,
)


sc.pl.umap(
    PRJNA779749_counts,
    color=["clusters"],
    title=f"UMAP: supervised density, PRJNA779749 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    add_outline=True,
    legend_loc="on data",
    legend_fontsize=12,
    legend_fontoutline=2,
    frameon=False,
    palette="nipy_spectral",
    show=True,
    save="-clusters" + save_PRJNA779749,
)


sc.pl.umap(
    PRJNA779749_counts,
    color=[g for g in npep if g in PRJNA779749_counts.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep" + save_PRJNA779749,
)


sc.pl.umap(
    PRJNA779749_counts,
    color=[g for g in npr if g in PRJNA779749_counts.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr" + save_PRJNA779749,
)


sc.pl.umap(
    PRJNA779749_counts,
    color=[g for g in nmr if g in PRJNA779749_counts.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-nmr" + save_PRJNA779749,
)


sc.pl.umap(
    PRJNA779749_counts,
    color=[g for g in genes_embed if g in PRJNA779749_counts.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgen" + save_PRJNA779749,
)


outlier_scores = sklearn.neighbors.LocalOutlierFactor(
    n_neighbors=ks, contamination=0.1
).fit_predict(PRJNA779749_counts.obsm["X_umap"])


PRJNA779749_outlying = PRJNA779749_counts[outlier_scores == -1]
PRJNA779749_outlying.shape


PRJNA779749_outlying


sc.pl.umap(
    PRJNA779749_outlying,
    color=[g for g in genes_embed if g in PRJNA779749_counts.var_names],
    frameon=False,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save=False,
)


sc.pl.umap(
    PRJNA779749_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA779749 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_outlying" + save_PRJNA779749,
)


sc.pl.umap(
    PRJNA779749_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA779749 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_outlying" + save_PRJNA779749,
)


PRJNA779749_wo_outlying = PRJNA779749_counts[outlier_scores != -1]
PRJNA779749_wo_outlying.shape


sc.pl.umap(
    PRJNA779749_wo_outlying,
    color=[g for g in genes_embed if g in PRJNA779749_counts.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-adgene_wo_outlying" + save_PRJNA779749,
)


sc.pl.umap(
    PRJNA779749_wo_outlying,
    color=[g for g in npep if g in PRJNA779749_counts.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npep_wo_outlying" + save_PRJNA779749,
)


sc.pl.umap(
    PRJNA779749_wo_outlying,
    color=[g for g in npr if g in PRJNA779749_counts.var_names],
    frameon=False,
    legend_fontsize=12,
    legend_fontoutline=2,
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-npr_wo_outlying" + save_PRJNA779749,
)


sc.pl.umap(
    PRJNA779749_wo_outlying,
    color=["learning"],
    title=f"UMAP: supervised density outlying, PRJNA779749 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-learning_split_wo_outlying" + save_PRJNA779749,
)


sc.pl.umap(
    PRJNA779749_wo_outlying,
    color=["clusters"],
    title=f"UMAP: supervised density outlying, PRJNA779749 \n(amb.FPR={cb_fpr}, k={k}, MSP={connectivity_model}, \nmetric={metric}, n_genes_excluded={signature})",
    legend_loc="on data",
    frameon=False,
    add_outline=True,
    legend_fontsize=12,
    legend_fontoutline=2,
    palette="nipy_spectral",
    show=True,
    size=40,
    na_color="#9A9FB080",
    save="-clusters_wo_outlying" + save_PRJNA779749,
)


PRJNA779749_counts.obs.clusters = PRJNA779749_counts.obs.clusters.astype(str)
PRJNA779749_counts.obs["clusters"] = pd.Categorical(PRJNA779749_counts.obs["clusters"])
features = np.array(PRJNA779749_counts.var_names)


# # Perform differential expression
# if os.path.isfile(de_picklefile):
#     de = pickle.load(open(de_picklefile, "rb"))
# else:
#     de = {}
# start = time.time()
# for i, (c1, c2) in enumerate(
#     itertools.combinations(np.unique(predictions_PRJNA779749), 2)
# ):
#     p_t = ttest_ind(
#         X_train[membership == c1].todense(), X_train[membership == c2].todense()
#     )[1]
#     p_t[np.isnan(p_t)] = 1
#     y = np.array(X_test[predictions_PRJNA779749 == c1].todense())
#     z = np.array(X_test[predictions_PRJNA779749 == c2].todense())
#     a = np.array(svm_PRJNA779749.coef_[i].todense()).reshape(-1)
#     p_tn = tn.tn_test(
#         y,
#         z,
#         a=a,
#         b=svm_PRJNA779749.intercept_[i],
#         learning_rate=1.0,
#         eps=1e-2,
#         verbose=verbose,
#         num_iters=100000,
#         num_cores=snakemake.threads,
#     )
#     de[(c1, c2)] = (p_t, p_tn)
#     print("c1: %5s\tc2: %5s\ttime elapsed: %.2fs" % (c1, c2, time.time() - start))
#     pickle.dump(de, open(de_picklefile, "wb"))


# # visualize results
# ngenes = 30
# ind = np.arange(ngenes)

# for c1, c2 in sorted(de):
#     p_t, p_tn = de[(c1, c2)]
#     gene_inds = np.argsort(p_t)[:ngenes]
#     gene_names = features[gene_inds]

#     fig, ax = plt.subplots(figsize=(3, 1.5))
#     rects = ax.bar(
#         ind - 0.2, -np.log10(p_t[gene_inds]), 0.35, align="center", label=r"$t$-test"
#     )
#     rects = ax.bar(
#         ind - 0.2 + 0.3,
#         -np.log10(p_tn[gene_inds]),
#         0.35,
#         align="center",
#         label="TN test",
#     )
#     ax.set_ylabel("-log($p$)")
#     xticks_pos = [0.65 * patch.get_width() + patch.get_xy()[0] - 0.2 for patch in rects]
#     plt.xticks(np.array(xticks_pos), gene_names, rotation=90)
#     plt.legend()
#     plt.title("Cluster %s v cluster %s" % (c1, c2))
#     plt.show()


# plt.savefig(
#     os.path.join(
#         PLOTS_DIR,
#         f"astrocytes-predictions-PRJNA779749-msp_{connectivity_model}-metric_{metric}-k_{k}-sign_{signature}-amb_{cb_fpr}-tntest_results.pdf",
#     )
# )
